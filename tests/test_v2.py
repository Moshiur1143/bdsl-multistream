from pathlib import Path
import sys,json,time
import pandas as pd,torch
ROOT=Path(r'F:\ML');sys.path.insert(0,str(ROOT))
from src.model_v2 import MultiStreamSignModelV2
from src.data_v2 import FeatureDatasetV2
from torch.utils.data import DataLoader
torch.manual_seed(42);torch.set_num_threads(8)
df=pd.read_csv(ROOT/'manifests/feature_manifest_v2.csv');train=df[df.split=='train']
rows=pd.concat([train[train.class_id==c].head(2) for c in range(8)])
x,y=next(iter(DataLoader(FeatureDatasetV2(rows,ROOT),batch_size=16)))
x={k:v.cuda() for k,v in x.items()};y=y.cuda();model=MultiStreamSignModelV2().cuda()
opt=torch.optim.AdamW(model.parameters(),lr=.001);scaler=torch.amp.GradScaler('cuda')
losses=[]
for step in range(50):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        logits=model(**x);loss=torch.nn.functional.cross_entropy(logits,y)
    assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    scaler.step(opt);scaler.update();losses.append(loss.item())
assert losses[-1]<losses[0]*.5,(losses[0],losses[-1])
result=dict(test='16 training clips only, optimization sanity check',initial_loss=losses[0],final_loss=losses[-1],parameters=sum(p.numel() for p in model.parameters()),not_validation_accuracy=True)
(ROOT/'results/v2_sanity.json').write_text(json.dumps(result,indent=2));print(result)
