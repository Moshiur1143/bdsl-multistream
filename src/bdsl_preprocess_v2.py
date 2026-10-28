"""SignBD synchronized masked features. No train/test fitting occurs here."""
from pathlib import Path
from itertools import permutations
import json, hashlib, os, time
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

CONFIG = dict(version='2.1-local', face_detector='short_range_upper_body_roi', frames=32, max_side=960, face_size=96,
              sampling='uniform_actual_frames', slots='pose_matched_left_right',
              graph='left21_right21_pose33_xy', missing='zero_plus_explicit_mask')

def sample_video(path):
    cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened(): raise ValueError('Cannot open video: '+str(path))
    n=0
    while cap.grab(): n+=1
    cap.release()
    if n<2 or not np.isfinite(fps) or fps<=0: raise ValueError('Invalid video frames/FPS')
    ids=np.rint(np.linspace(0,n-1,32)).astype(np.int32); wanted=set(ids.tolist()); cache={}
    cap=cv2.VideoCapture(str(path))
    try:
        for i in range(n):
            if not cap.grab(): break
            if i in wanted:
                ok,bgr=cap.retrieve()
                if ok:
                    h,w=bgr.shape[:2]; scale=min(1,960/max(h,w))
                    if scale<1: bgr=cv2.resize(bgr,(round(w*scale),round(h*scale)))
                    cache[i]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    finally: cap.release()
    images=[cache.get(int(i)) for i in ids]
    mask=np.array([a is not None for a in images])
    if mask.sum()<16: raise ValueError('Fewer than 16 decoded sample positions')
    return images,ids,mask,ids.astype(np.float32)/fps,n

def mpi(a): return mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(a))

class Extractor:
    def __init__(self,models):
        models=Path(models)
        self.config={**CONFIG,'mediapipe':mp.__version__,'opencv':cv2.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'model_sha256':{n:hashlib.sha256((models/n).read_bytes()).hexdigest() for n in ['hand.task','pose.task','face_short.tflite']}}
        self.key=hashlib.sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest()[:16]
        base=lambda n: python.BaseOptions(model_asset_path=str(models/n))
        self.hand=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=base('hand.task'),num_hands=2))
        self.pose=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=base('pose.task'),num_poses=1))
        self.face=vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=base('face_short.tflite')))
    def close(self):
        for detector in [self.hand,self.pose,self.face]: detector.close()
    def process(self,path):
        images,ids,mask,times,n=sample_video(path)
        return self.process_samples(images,ids,mask,times,n)
    def process_frame(self,rgb):
        h,w=rgb.shape[:2]; scale=min(1,960/max(h,w))
        if scale<1: rgb=cv2.resize(rgb,(round(w*scale),round(h*scale)))
        mask=np.zeros(32,bool); mask[0]=True
        arrays,quality,preview=self.process_samples([rgb]+[None]*31,np.arange(32),mask,np.arange(32,dtype=np.float32),1)
        return {k:v[0].copy() for k,v in arrays.items()}
    def process_samples(self,images,ids,mask,times,n):
        start=time.time()
        a=dict(graph_xy=np.zeros((32,75,2),np.float32),graph_mask=np.zeros((32,75),bool),
          hand_local_xyz=np.zeros((32,2,21,3),np.float32),hand_mask=np.zeros((32,2),bool),
          assignment_source=np.zeros((32,2),np.uint8),hand_score=np.zeros((32,2),np.float32),
          face_rgb=np.zeros((32,96,96,3),np.uint8),face_mask=np.zeros(32,bool),
          face_box=np.zeros((32,4),np.float32),pose_visibility=np.zeros((32,33),np.float32),
          frame_mask=mask,frame_indices=ids,timestamps=times,
          unique_frame_mask=mask & np.r_[True,np.diff(ids)>0])
        preview=None
        for t,rgb in enumerate(images):
            if rgb is None: continue
            if t and ids[t]==ids[t-1]:
                for k in ['graph_xy','graph_mask','hand_local_xyz','hand_mask','assignment_source','hand_score','face_rgb','face_mask','face_box','pose_visibility']: a[k][t]=a[k][t-1]
                continue
            h,w=rgb.shape[:2]; pr=self.pose.detect(mpi(rgb)); xy=None; pm=None
            x1,y1,x2,y2=0,0,w,h
            if pr.pose_landmarks:
                p=pr.pose_landmarks[0]; xy=np.array([[v.x,v.y] for v in p],np.float32)
                pm=np.array([min(v.visibility,v.presence)>=.5 for v in p])
                a['graph_xy'][t,42:]=xy*pm[:,None]; a['graph_mask'][t,42:]=pm
                a['pose_visibility'][t]=[v.visibility for v in p]
                upper=[j for j in [0,*range(11,25)] if pm[j]]
                if len(upper)>=4:
                    pts=xy[upper]*[w,h]; lo=pts.min(0); hi=pts.max(0); pad=.2*max(hi-lo)
                    x1,y1=np.maximum(0,lo-pad).astype(int); x2,y2=np.minimum([w,h],hi+pad).astype(int)
            if x2<=x1 or y2<=y1: x1,y1,x2,y2=0,0,w,h
            crop=rgb[y1:y2,x1:x2]; hr=self.hand.detect(mpi(crop)); candidates=[]
            for j,hand in enumerate(hr.hand_landmarks):
                raw=np.array([[v.x,v.y,v.z] for v in hand],np.float32)
                global_xy=(raw[:,:2]*[x2-x1,y2-y1]+[x1,y1])/[w,h]
                local=raw-raw[0]; local[:,1]*=(y2-y1)/(x2-x1)
                local/=max(float(np.linalg.norm(local[9,:2])),1e-6)
                label=hr.handedness[j][0]; candidates.append((global_xy,local,label))
            def cost(slots):
                total=0
                for item,slot in zip(candidates,slots):
                    wrist=15+slot
                    if xy is not None and pm[wrist]: total+=np.linalg.norm((item[0][0]-xy[wrist])*[w/h,1])
                    else: total+=.15 if item[2].category_name==['Left','Right'][slot] else .35
                return total
            if candidates:
                slots=min(permutations(range(2),len(candidates)),key=cost)
                for (points,local,label),slot in zip(candidates,slots):
                    sl=slice(slot*21,(slot+1)*21)
                    a['graph_xy'][t,sl]=points; a['graph_mask'][t,sl]=True
                    a['hand_local_xyz'][t,slot]=local; a['hand_mask'][t,slot]=True
                    a['hand_score'][t,slot]=label.score
                    a['assignment_source'][t,slot]=1 if xy is not None and pm[15+slot] else 2
            detections=self.face.detect(mpi(crop)).detections
            if detections:
                def center(d):
                    b=d.bounding_box; return np.array([(x1+b.origin_x+b.width/2)/w,(y1+b.origin_y+b.height/2)/h])
                nose=xy[0] if xy is not None and pm[0] else None
                d=min(detections,key=lambda d:np.linalg.norm(center(d)-nose)) if nose is not None else max(detections,key=lambda d:d.bounding_box.width*d.bounding_box.height)
                b=d.bounding_box; cx,cy=center(d)*[w,h]; side=1.35*max(b.width,b.height)
                l,top=np.maximum([0,0],[cx-side/2,cy-side/2]).astype(int)
                r,bot=np.minimum([w,h],[cx+side/2,cy+side/2]).astype(int)
                if r>l and bot>top and (nose is None or np.linalg.norm(center(d)-nose)<.2):
                    a['face_rgb'][t]=cv2.resize(rgb[top:bot,l:r],(96,96))
                    a['face_mask'][t]=True; a['face_box'][t]=np.array([l,top,r,bot])/[w,h,w,h]
            if preview is None or t==16:
                preview=rgb.copy()
                for point,valid in zip(a['graph_xy'][t,:42],a['graph_mask'][t,:42]):
                    if valid: cv2.circle(preview,tuple((point*[w,h]).astype(int)),2,(0,255,0),-1)
        valid=a['unique_frame_mask']; denom=max(1,valid.sum())
        quality=dict(actual_frames=n,decoded=int(mask.sum()),unique=int(valid.sum()),seconds=time.time()-start,
          any_hand=float((a['hand_mask'].any(1)&valid).sum()/denom),
          both_hands=float((a['hand_mask'].all(1)&valid).sum()/denom),
          face=float((a['face_mask']&valid).sum()/denom))
        return a,quality,preview

def cache_path(root,row): return Path(root)/row['split']/str(row['class_id'])/(Path(row['path']).stem+'.npz')
def valid_cache(path,key,row):
    try:
        with np.load(path,allow_pickle=False) as z:
            return (str(z['config_id'])==key and str(z['relative_path'])==row['relative_path']
              and int(z['class_id'])==row['class_id'] and z['graph_xy'].shape==(32,75,2)
              and z['face_rgb'].shape==(32,96,96,3) and np.isfinite(z['graph_xy']).all()
              and int(z['frame_mask'].sum())>=16)
    except (OSError,ValueError,KeyError,EOFError): return False

def save_feature(path,a,q,key,row):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**a,config_id=key,class_id=int(row['class_id']),
          relative_path=row['relative_path'],signer=row['signer'],split=row['split'],quality_json=json.dumps(q))
    os.replace(tmp,path)
    if not valid_cache(path,key,row): raise ValueError('Cache verification failed')

def extract_all(df,models,root,results):
    from tqdm.auto import tqdm
    results=Path(results); results.mkdir(parents=True,exist_ok=True)
    ex=Extractor(models); start=time.time(); done=skip=failed=0
    (results/'feature_config_v2.json').write_text(json.dumps(ex.config,indent=2))
    try:
        for row in tqdm(df.to_dict('records'),desc='Persistent features'):
            target=cache_path(root,row)
            if valid_cache(target,ex.key,row): skip+=1
            else:
                try:
                    arrays,q,_=ex.process(row['path']); save_feature(target,arrays,q,ex.key,row); done+=1
                    event={**row,**q,'status':'ok'}
                except (ValueError,RuntimeError) as err:
                    failed+=1; event={**row,'status':'failed','error':str(err)}; print(event,flush=True)
                with (results/'extraction_log_v2.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
            status=dict(total=len(df),new=done,cached=skip,failed=failed,elapsed_seconds=time.time()-start,config_id=ex.key)
            tmp=results/'progress_v2.tmp'; tmp.write_text(json.dumps(status,indent=2)); os.replace(tmp,results/'progress_v2.json')
            if failed>=10: raise RuntimeError('Stopped after 10 failures; inspect log before resuming')
    finally: ex.close()
    print(status,flush=True)
    return status


