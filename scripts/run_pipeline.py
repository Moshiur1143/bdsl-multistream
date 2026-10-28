"""Run persistent data preparation and three validation-selected training experiments."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'results/pipeline_status.json'

def status(stage,state,**extra):
    payload=dict(stage=stage,state=state,updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),**extra)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(STATUS)
    print(json.dumps(payload),flush=True)

def run(stage,script,*args):
    status(stage,'running')
    subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,check=True)
    status(stage,'complete')

def main():
    lock=ROOT/'work/pipeline.lock';lock.parent.mkdir(exist_ok=True)
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise SystemExit('Pipeline lock exists. Check the running process before restarting.')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        run('download','scripts/download_dataset.py')
        run('dataset preparation','scripts/prepare_dataset.py')
        run('preprocessing pilot','scripts/test_preprocessing.py','--pilot')
        run('feature extraction','scripts/extract_features.py')
        for mode in ('graph','face','fusion'):
            resume=['--resume'] if (ROOT/f'models/{mode}/last.pt').exists() else []
            run(mode+' training','scripts/train.py','--mode',mode,*resume)
            run(mode+' validation','scripts/train.py','--mode',mode,'--evaluate','val')
        scores={m:json.loads((ROOT/f'results/{m}/val_metrics.json').read_text())['macro_f1'] for m in ('graph','face','fusion')}
        winner=max(scores,key=scores.get)
        (ROOT/'results/model_selection.json').write_text(json.dumps(dict(selected=winner,validation_macro_f1=scores,test_evaluated=False),indent=2))
        status('training pipeline','complete',selected_model=winner,next='Review validation results, evaluate selected checkpoint on test, and review translations before webcam demo.')
    except BaseException as error:
        status('pipeline','failed',error=str(error));traceback.print_exc();raise
    finally:
        lock.unlink(missing_ok=True)
if __name__=='__main__':main()
