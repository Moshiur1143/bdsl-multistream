"""Local-only webcam UI. Requires a trained checkpoint and reviewed translations."""
import argparse,json,sys,time,threading
from pathlib import Path
from collections import deque
import cv2,numpy as np,torch
from flask import Flask,request,jsonify,send_from_directory
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from bilingual_labels import BilingualLabels
from bdsl_preprocess_v2 import Extractor
from data import normalize
from live_geometry import align_live_graph
from model import MultiStreamSignModel

def create_app(checkpoint,threshold=.8,review_mode=False):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    saved=torch.load(checkpoint,map_location=device,weights_only=False)
    labels=BilingualLabels(ROOT/'manifests/label_map.json',ROOT/'manifests/translations.json',require_reviewed=not review_mode)
    assert saved['label_map']==json.loads((ROOT/'manifests/label_map.json').read_text(encoding='utf-8-sig'))
    mode=saved['args']['mode']
    if mode in ('graph_v2','fusion_v2'):
        from src.model_v2 import MultiStreamSignModelV2
        from src.data_v2 import normalize_v2
        model=MultiStreamSignModelV2(len(labels.by_id),mode,pretrained_face=False).to(device).eval()
        normalize_fn=normalize_v2
    else:
        model=MultiStreamSignModel(len(labels.by_id),mode).to(device).eval()
        normalize_fn=normalize
    model.load_state_dict(saved['model'])
    extractor=Extractor(ROOT/'models/detectors')
    assert saved['feature_config']==extractor.config,'Preprocessing differs from the trained checkpoint'
    print('Warming up the model before accepting camera frames...',flush=True)
    channels=10 if mode in ('graph_v2','fusion_v2') else 5
    with torch.inference_mode():
        model(graph=torch.zeros(1,32,75,channels,device=device),
              graph_mask=torch.ones(1,32,75,dtype=torch.bool,device=device),
              face=torch.zeros(1,32,3,96,96,device=device),
              face_mask=torch.ones(1,32,dtype=torch.bool,device=device),
              frame_mask=torch.ones(1,32,dtype=torch.bool,device=device))
    if device=='cuda':torch.cuda.synchronize()
    app=Flask(__name__,static_folder=str(ROOT/'web'));app.config['MAX_CONTENT_LENGTH']=2*1024*1024
    buffer=deque(maxlen=32);recent=deque(maxlen=3);lock=threading.Lock();last=[0.];last_shape=[None]
    @app.before_request
    def local_only():
        # Prevent a foreign web page from posting webcam data to this localhost service.
        origin=request.headers.get('Origin')
        if origin and origin not in ('http://127.0.0.1:5000','http://localhost:5000'):return 'Forbidden origin',403
    @app.get('/status')
    def status():return jsonify(review_mode=review_mode,pending_translations=len(labels.pending),model=saved['args']['mode'])
    @app.get('/')
    def index():return send_from_directory(ROOT/'web','index.html')
    @app.post('/reset')
    def reset():
        with lock:buffer.clear();recent.clear();last[0]=0.
        return jsonify(ok=True)
    @app.post('/frame')
    def frame():
        if not lock.acquire(blocking=False):return jsonify(error='busy'),429
        try:
            start=time.perf_counter()
            if start-last[0]>2:buffer.clear();recent.clear()
            # Inactivity is measured after the previous request finishes.
            bgr=cv2.imdecode(np.frombuffer(request.get_data(),np.uint8),cv2.IMREAD_COLOR)
            if bgr is None or max(bgr.shape[:2])>1920:return jsonify(error='Invalid frame'),400
            if last_shape[0]!=bgr.shape[:2]:
                buffer.clear();recent.clear();last_shape[0]=bgr.shape[:2]
            item=extractor.process_frame(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            item=align_live_graph(item,bgr.shape[1],bgr.shape[0]);buffer.append(item)
            result=labels.display();confidence=0.
            top3=[];food_confidence=None;food_rank=None;visible=False
            reason='Collecting 32 frames'
            hands=int(np.asarray(item['hand_mask']).sum())
            face_detected=bool(np.asarray(item['face_mask']).any())
            if len(buffer)==32:
                arrays={k:np.stack([r[k] for r in buffer]) for k in item}
                arrays['frame_indices']=np.arange(32);arrays['unique_frame_mask']=np.ones(32,bool)
                batch={k:torch.from_numpy(v.copy()).unsqueeze(0).to(device) for k,v in normalize_fn(arrays).items()}
                with torch.no_grad():probs=model(**batch).softmax(-1)[0]
                food_id=next(i for i,row in labels.by_id.items() if row['gloss']=='khabar')
                food_confidence=float(probs[food_id]);food_rank=int((probs>probs[food_id]).sum())+1
                scores,indices=probs.topk(3)
                top3=[dict(**labels.display(int(i),accepted=True),confidence=float(p)) for p,i in zip(scores,indices)]
                confidence,cid=probs.max(0);confidence=float(confidence);cid=int(cid)
                visible=bool(np.stack([r['hand_mask'] for r in list(buffer)[-4:]]).any(1).all())
                recent.append(cid if confidence>=threshold and visible else None)
                reason=('Hand not consistently detected in last 4 frames' if not visible else
                        'Confidence below acceptance threshold' if confidence<threshold else
                        'Waiting for 3 consistent predictions')
                if len(recent)==3 and recent[0] is not None and len(set(recent))==1:result=labels.display(cid,accepted=True);reason="Accepted"
            return jsonify(**result,diagnostics=dict(hands_detected=hands,face_detected=face_detected,
                recent_hand_visible=visible,top3=top3,khabar_confidence=food_confidence,khabar_rank=food_rank,
                threshold=threshold,reason=reason,geometry='training-aspect-16:9',input_width=bgr.shape[1],input_height=bgr.shape[0]),confidence=confidence,buffer_frames=len(buffer),processing_ms=round(1000*(time.perf_counter()-start),1))
        finally:
            last[0]=time.perf_counter()
            lock.release()
    return app,extractor

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['graph','face','fusion','graph_v2','fusion_v2'],default='graph');p.add_argument('--threshold',type=float,default=.8);p.add_argument('--review-mode',action='store_true');a=p.parse_args()
    checkpoint=ROOT/f'models/{a.mode}/best.pt'
    if not checkpoint.exists():raise SystemExit('No trained model yet. Finish extraction and training first.')
    if not 0<a.threshold<1:raise SystemExit('Threshold must be between 0 and 1')
    app,extractor=create_app(checkpoint,a.threshold,a.review_mode)
    print('Open http://127.0.0.1:5000. Threshold is provisional; calibrate on separate idle/unseen-sign data.')
    try:app.run(host='127.0.0.1',port=5000,debug=False,threaded=True)
    finally:extractor.close()




