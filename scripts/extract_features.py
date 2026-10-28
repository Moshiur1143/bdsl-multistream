"""Local extraction and feature manifest audit. Stored paths remain portable."""
import sys,json,hashlib,urllib.request
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import bdsl_preprocess_v2 as prep
URLS={
'hand.task':'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
'pose.task':'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
'face_short.tflite':'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'}

def main():
    models=ROOT/'models/detectors';models.mkdir(parents=True,exist_ok=True)
    for name,url in URLS.items():
        target=models/name
        if not target.exists():
            tmp=target.with_suffix('.part');urllib.request.urlretrieve(url,tmp);tmp.replace(target)
    df=pd.read_csv(ROOT/'manifests/all_videos.csv');df['path']=df.path.map(lambda p:str(ROOT/p))
    ordered=pd.concat([df[df.split==s].sort_values(['class_id','signer','view']) for s in ('train','val','test')])
    report=prep.extract_all(ordered,models,ROOT/'features/v2',ROOT/'results')
    if report['failed']:raise RuntimeError('Resolve extraction failures before training')
    config=json.loads((ROOT/'results/feature_config_v2.json').read_text())
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    rows=[]
    for row in ordered.to_dict('records'):
        target=prep.cache_path(ROOT/'features/v2',row)
        assert prep.valid_cache(target,key,row),target
        row['path']=Path(row['path']).relative_to(ROOT).as_posix()
        row['feature_path']=target.relative_to(ROOT).as_posix();row['config_id']=key
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'manifests/feature_manifest_v2.csv',index=False)
    print('All 6000 features verified. Ready to train.',flush=True)
if __name__=='__main__':main()

