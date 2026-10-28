"""Train on frozen signer splits; checkpoint selection uses validation macro-F1."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model import MultiStreamSignModel
from src.data import FeatureDataset

def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def load_manifest():
    p=ROOT/'manifests/feature_manifest_v2.csv'
    if not p.exists():raise RuntimeError('Complete feature extraction and audit first.')
    df=pd.read_csv(p); labels=json.loads((ROOT/'manifests/label_map.json').read_text(encoding='utf-8-sig'))
    if len(labels)<150 or sorted(labels.values())!=list(range(len(labels))):raise ValueError('Invalid class map')
    assert len(df)==6000 and df.relative_path.nunique()==6000
    assert df.config_id.nunique()==1,'Mixed preprocessing configurations'
    groups={s:set(df[df.split==s].signer) for s in ('train','val','test')}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    for split in groups:assert set(df[df.split==split].class_id)==set(labels.values())
    for row in df.itertuples():
        assert labels[row.gloss]==row.class_id
        assert (ROOT/row.feature_path).is_file(),row.feature_path
    return df,labels,hashlib.sha256(p.read_bytes()).hexdigest()

def loader(df,split,batch,augment=False):
    return DataLoader(FeatureDataset(df[df.split==split],ROOT,augment),batch_size=batch,
                      shuffle=augment,num_workers=0,pin_memory=torch.cuda.is_available())

def move(x,device):return {k:v.to(device,non_blocking=True) for k,v in x.items()}

@torch.no_grad()
def evaluate(model,batches,device,num_classes):
    model.eval();truth=[];pred=[];top5=[];loss_sum=0;count=0
    for x,y in batches:
        x=move(x,device);y=y.to(device)
        logits=model(**x);loss_sum+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item();count+=len(y)
        truth.extend(y.cpu().tolist());pred.extend(logits.argmax(-1).cpu().tolist())
        top5.extend((logits.topk(min(5,num_classes),dim=1).indices==y[:,None]).any(1).cpu().tolist())
    return dict(loss=loss_sum/max(count,1),accuracy=accuracy_score(truth,pred),
                macro_f1=f1_score(truth,pred,labels=list(range(num_classes)),average='macro',zero_division=0),
                top5=float(np.mean(top5))),truth,pred

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['graph','face','fusion'],default='graph')
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--patience',type=int,default=12)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',action='store_true')
    p.add_argument('--evaluate',choices=['val','test']);args=p.parse_args()
    seed_all(args.seed);torch.set_num_threads(8)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df,labels,manifest_hash=load_manifest(); n=len(labels)
    out=ROOT/'results'/args.mode;out.mkdir(parents=True,exist_ok=True)
    checkpoints=ROOT/'models'/args.mode;checkpoints.mkdir(parents=True,exist_ok=True)
    model=MultiStreamSignModel(n,args.mode).to(device)
    print(f'{args.mode}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}',flush=True)
    if args.evaluate:
        checkpoint=torch.load(checkpoints/'best.pt',map_location=device,weights_only=False)
        assert checkpoint['label_map']==labels and checkpoint['manifest_hash']==manifest_hash
        model.load_state_dict(checkpoint['model'])
        metrics,y,pred=evaluate(model,loader(df,args.evaluate,args.batch_size),device,n)
        names=[g for g,i in sorted(labels.items(),key=lambda item:item[1])]
        payload={**metrics,'split':args.evaluate,'checkpoint_epoch':checkpoint['epoch'],
            'classification_report':classification_report(y,pred,labels=list(range(n)),target_names=names,zero_division=0,output_dict=True)}
        (out/f'{args.evaluate}_metrics.json').write_text(json.dumps(payload,indent=2))
        pd.DataFrame(confusion_matrix(y,pred,labels=list(range(n))),index=names,columns=names).to_csv(out/f'{args.evaluate}_confusion.csv')
        pd.DataFrame({'truth':y,'prediction':pred}).to_csv(out/f'{args.evaluate}_predictions.csv',index=False)
        print(metrics);return
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start=0;best=-1.;stale=0;history=[]
    if args.resume:
        checkpoint=torch.load(checkpoints/'last.pt',map_location='cpu',weights_only=False)
        assert checkpoint['label_map']==labels and checkpoint['manifest_hash']==manifest_hash
        assert checkpoint['args']['epochs']==args.epochs,'Keep epoch schedule unchanged on resume'
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
        start=checkpoint['epoch']+1;best=checkpoint['best'];stale=checkpoint['stale'];history=checkpoint['history']
        torch.set_rng_state(checkpoint['torch_rng']);np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['random_rng'])
        if device.type=='cuda' and checkpoint['cuda_rng'] is not None:torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    elif (checkpoints/'last.pt').exists():raise RuntimeError('Existing run: use --resume or preserve/rename it before a new experiment.')
    training=loader(df,'train',args.batch_size,True);validation=loader(df,'val',args.batch_size)
    for epoch in range(start,args.epochs):
        tick=time.time();model.train();loss_sum=0;count=0
        for step,(x,y) in enumerate(training):
            x=move(x,device);y=y.to(device);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits=model(**x);loss=torch.nn.functional.cross_entropy(logits,y,label_smoothing=.05)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update();loss_sum+=loss.item()*len(y);count+=len(y)
            if step%50==0:print(f'Epoch {epoch+1}, batch {step}/{len(training)}, loss {loss.item():.4f}',flush=True)
        metrics,_,_=evaluate(model,validation,device,n);scheduler.step()
        improved=metrics['macro_f1']>best
        if improved:best=metrics['macro_f1'];stale=0
        else:stale+=1
        history.append(dict(epoch=epoch+1,train_loss=loss_sum/count,seconds=time.time()-tick,**metrics))
        payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
            epoch=epoch,best=best,stale=stale,history=history,args=vars(args),label_map=labels,manifest_hash=manifest_hash,
            feature_config=json.loads((ROOT/'results/feature_config_v2.json').read_text()),
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),random_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        atomic_save(payload,checkpoints/'last.pt')
        if improved:atomic_save(payload,checkpoints/'best.pt')
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        print(history[-1],flush=True)
        if stale>=args.patience:print('Early stopping on validation macro-F1',flush=True);break
    print('Training finished. Test split has NOT been used for model selection.',flush=True)
if __name__=='__main__':main()
