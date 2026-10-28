from pathlib import Path
import subprocess,sys,json,time
ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'results/v2_status.json'
def record(stage,state,**kwargs):
    STATUS.write_text(json.dumps(dict(stage=stage,state=state,updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),**kwargs),indent=2))
try:
    for mode in ('graph_v2','fusion_v2'):
        record(mode,'training')
        args=['--resume'] if (ROOT/f'models/{mode}/last.pt').exists() else []
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/train_v2.py'),'--mode',mode,*args],cwd=ROOT,check=True)
        record(mode,'validation')
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts/train_v2.py'),'--mode',mode,'--evaluate','val'],cwd=ROOT,check=True)
    modes=['graph','fusion','graph_v2','fusion_v2']
    metrics={m:json.loads((ROOT/f'results/{m}/val_metrics.json').read_text())['macro_f1'] for m in modes}
    record('comparison','complete',validation_macro_f1=metrics,selected=max(metrics,key=metrics.get),test_reused=False)
except BaseException as error:
    record('experiment','failed',error=str(error));raise
