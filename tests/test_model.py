import json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model import MultiStreamSignModel

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'; results={}
    for mode in ('graph','face','fusion'):
        model=MultiStreamSignModel(200,mode).to(device)
        x=dict(graph=torch.randn(2,32,75,5,device=device),graph_mask=torch.ones(2,32,75,dtype=torch.bool,device=device),
               face=torch.randn(2,32,3,96,96,device=device),face_mask=torch.ones(2,32,dtype=torch.bool,device=device),
               frame_mask=torch.ones(2,32,dtype=torch.bool,device=device))
        logits=model(**x); assert logits.shape==(2,200) and torch.isfinite(logits).all()
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,199],device=device));loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.eval();x['graph_mask'].zero_();x['face_mask'].zero_()
        with torch.no_grad():
            a=model(**x);x['graph'].fill_(900);x['face'].fill_(-900);b=model(**x)
        assert torch.allclose(a,b,atol=1e-5),'Missing-data values leaked into output'
        results[mode]=dict(parameters=sum(p.numel() for p in model.parameters()),forward_backward='passed',missing_mask_invariance='passed')
    result=dict(device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
                torch=torch.__version__,tests=results,note='Synthetic software checks only; no recognition accuracy measured.')
    (ROOT/'results/model_smoke_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
