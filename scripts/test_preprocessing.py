"""Local detector/decoder tests; --pilot adds 24 real training-only clips."""
from pathlib import Path
import sys,json,urllib.request,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bdsl_preprocess_v2 import Extractor,save_feature,cache_path,valid_cache
sys.path.insert(0,str(ROOT/'scripts'))
from extract_features import URLS

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    models=ROOT/'models/detectors';models.mkdir(parents=True,exist_ok=True)
    for name,url in URLS.items():
        target=models/name
        if not target.exists():
            tmp=target.with_suffix('.part');urllib.request.urlretrieve(url,tmp);tmp.replace(target)
    ex=Extractor(models)
    try:
        frame=ex.process_frame(np.zeros((480,640,3),dtype=np.uint8))
        assert frame['graph_xy'].shape==(75,2) and not frame['hand_mask'].any()
        assert not frame['face_mask'] and np.isfinite(frame['graph_xy']).all()
        report=dict(blank_frame_test='passed',config=ex.config)
        if args.pilot:
            import pandas as pd
            df=pd.read_csv(ROOT/'manifests/train.csv');labels=np.random.default_rng(42).choice(sorted(df.gloss.unique()),24,replace=False);rows=[]
            for i,label in enumerate(labels):
                row=df[(df.gloss==label)&(df.view==('c' if i%2==0 else 'f'))].sample(1,random_state=42+i).iloc[0].to_dict()
                row['path']=str(ROOT/row['path']);a,q,_=ex.process(row['path'])
                assert a['graph_xy'].shape==(32,75,2) and a['face_rgb'].shape==(32,96,96,3)
                assert q['decoded']>=16 and np.isfinite(a['hand_local_xyz']).all()
                target=cache_path(ROOT/'features/v2',row);save_feature(target,a,q,ex.key,row)
                assert valid_cache(target,ex.key,row)
                rows.append({**row,**q});print(f'Pilot {i+1}/24: {label}, {q}',flush=True)
            pd.DataFrame(rows).to_csv(ROOT/'results/local_pilot.csv',index=False)
            if np.mean([r['face'] for r in rows])<.7 or np.mean([r['any_hand'] for r in rows])<.6:
                raise RuntimeError('Pilot detection coverage is low; review local_pilot.csv before full extraction.')
        (ROOT/'results/preprocessing_test.json').write_text(json.dumps(report,indent=2))
        print('Preprocessing checks passed. These are not recognition accuracy results.',flush=True)
    finally:ex.close()
if __name__=='__main__':main()

