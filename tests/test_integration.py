"""Synthetic pipeline integration tests; never writes research accuracy results."""
from pathlib import Path
import sys,json
import cv2,numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from src import bdsl_preprocess_v2 as prep
from src.data import normalize,FeatureDataset
from src.bilingual_labels import BilingualLabels
from src.model import MultiStreamSignModel

def main():
    work=ROOT/'work/integration_test';work.mkdir(parents=True,exist_ok=True)
    video=work/'synthetic_blank.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(320,240))
    assert writer.isOpened()
    for _ in range(40):writer.write(np.zeros((240,320,3),np.uint8))
    writer.release();ex=prep.Extractor(ROOT/'models/detectors')
    try:arrays,q,_=ex.process(video)
    finally:ex.close()
    assert q['decoded']==32 and q['unique']==32 and not arrays['face_mask'].any()
    row=dict(path=str(video),relative_path='synthetic/blank.mp4',class_id=0,signer='synthetic',split='train')
    cache=work/'sample.npz';prep.save_feature(cache,arrays,q,ex.key,row)
    assert prep.valid_cache(cache,ex.key,row) and not prep.valid_cache(cache,'invalid',row)
    normalized=normalize(arrays);assert normalized['graph'].shape==(32,75,5)
    assert np.isfinite(normalized['graph']).all() and (normalized['graph']==0).all()
    rows=pd.DataFrame([dict(feature_path=cache.relative_to(ROOT).as_posix(),class_id=0)])
    x,y=FeatureDataset(rows,ROOT)[0];assert y==0 and x['face'].shape==(32,3,96,96)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # Check default batch size really fits, including backward/optimizer/AMP.
    model=MultiStreamSignModel(200,'fusion').to(device);optimizer=torch.optim.AdamW(model.parameters())
    batch={k:v[None].repeat(16,*([1]*v.ndim)).to(device) for k,v in x.items()}
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    with torch.autocast(device_type=device,enabled=device=='cuda'):
        loss=torch.nn.functional.cross_entropy(model(**batch),torch.zeros(16,dtype=torch.long,device=device))
    scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();assert torch.isfinite(loss)
    labels={f'test_{i}':i for i in range(200)}
    translations={g:dict(en='Synthetic test',bn='পরীক্ষা',reviewed=True) for g in labels}
    (work/'labels.json').write_text(json.dumps(labels),encoding='utf8')
    (work/'translations.json').write_text(json.dumps(translations,ensure_ascii=False),encoding='utf8')
    bilingual=BilingualLabels(work/'labels.json',work/'translations.json')
    assert bilingual.display(199,accepted=True)['bangla']=='পরীক্ষা' and bilingual.display()['class_id'] is None
    translations.pop('test_0');(work/'translations.json').write_text(json.dumps(translations),encoding='utf8')
    try:BilingualLabels(work/'labels.json',work/'translations.json')
    except ValueError:pass
    else:raise AssertionError('Incomplete translations accepted')
    result=dict(synthetic=True,decoder='passed',cache_readback='passed',normalization='passed',
      bilingual_coverage='passed',fusion_batch16_amp_optimizer='passed',device=device,
      peak_cuda_GiB=round(torch.cuda.max_memory_allocated()/1024**3,3) if device=='cuda' else None)
    (ROOT/'results/integration_test.json').write_text(json.dumps(result,indent=2));print(result)
if __name__=='__main__':main()
