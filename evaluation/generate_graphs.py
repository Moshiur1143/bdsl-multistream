"""Regenerate evaluation figures from recorded results; no training required."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
def save(name):
    plt.tight_layout(); plt.savefig(OUT/name,bbox_inches='tight'); plt.close()
def main():
    h=pd.read_csv(ROOT/'results/fusion_v2/history.csv')
    m=json.loads((ROOT/'results/fusion_v2/test_metrics.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(h.epoch,h.train_loss,label='Training loss'); ax[0].plot(h.epoch,h.loss,label='Validation loss'); ax[0].set_ylabel('Loss'); ax[0].legend()
    ax[1].plot(h.epoch,h.accuracy*100,label='Validation accuracy'); ax[1].plot(h.epoch,h.macro_f1*100,label='Validation macro-F1'); ax[1].set_ylabel('Percent'); ax[1].legend()
    for a in ax: a.set_xlabel('Epoch'); a.axvline(m['checkpoint_epoch']+1,color='gray',linestyle=':',label='Selected epoch'); a.grid(alpha=.2)
    fig.suptitle('Fusion V2 training history — dotted line: selected checkpoint'); save('training_curves.png')
    models=['graph','face','fusion','graph_v2','fusion_v2']; rows=[]
    for model in models:
        v=json.loads((ROOT/f'results/{model}/val_metrics.json').read_text()); rows.append({'model':model,'validation_accuracy':v['accuracy'],'validation_macro_f1':v['macro_f1']})
    comp=pd.DataFrame(rows); comp.to_csv(OUT/'validation_comparison.csv',index=False)
    x=np.arange(len(rows)); fig,ax=plt.subplots(figsize=(10,5)); ax.bar(x-.18,comp.validation_accuracy*100,.36,label='Accuracy'); ax.bar(x+.18,comp.validation_macro_f1*100,.36,label='Macro-F1'); ax.set_xticks(x,models); ax.set_ylim(0,100); ax.set_ylabel('Percent'); ax.set_title('Recorded validation results — different training configurations'); ax.legend(); save('validation_comparison.png')
    cm=pd.read_csv(ROOT/'results/fusion_v2/test_confusion.csv',index_col=0); p=pd.read_csv(ROOT/'results/fusion_v2/test_predictions.csv'); labels=json.loads((ROOT/'manifests/label_map.json').read_text(encoding='utf-8-sig')); names=[g for g,i in sorted(labels.items(),key=lambda x:x[1])]
    assert list(cm.index)==names and list(cm.columns)==names
    rebuilt=np.zeros((200,200),dtype=int)
    for t,v in p[['truth','prediction']].itertuples(index=False,name=None): rebuilt[t,v]+=1
    assert np.array_equal(cm.values,rebuilt) and rebuilt.sum()==1200
    correct=int(np.trace(rebuilt)); assert abs(correct/1200-m['accuracy'])<1e-9
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cm.values/np.maximum(cm.values.sum(axis=1,keepdims=True),1),vmin=0,vmax=1,cmap='Blues'); fig.colorbar(im,ax=ax,label='Fraction within true class'); ax.set_xlabel('Predicted class ID'); ax.set_ylabel('True class ID'); ax.set_title('Fusion V2: 200-class test confusion matrix'); save('confusion_matrix.png')
    pairs=sorted([(int(rebuilt[i,j]),names[i],names[j]) for i in range(200) for j in range(200) if i!=j and rebuilt[i,j]],key=lambda x:(-x[0],x[1],x[2])); errors=pd.DataFrame(pairs,columns=['count','true_gloss','predicted_gloss']); errors.to_csv(OUT/'confusion_pairs.csv',index=False)
    top=errors.head(15).iloc[::-1]; fig,ax=plt.subplots(figsize=(10,7)); ax.barh(top.true_gloss+' → '+top.predicted_gloss,top['count'],color='#db7856'); ax.set_xlabel('Misclassified test clips'); ax.set_title('Most frequent directed confusions (ties sorted alphabetically)'); save('top_confusions.png')
    per=pd.DataFrame([dict(gloss=g,**m['classification_report'][g]) for g in names]); per.to_csv(OUT/'per_class_metrics.csv',index=False)
    low=per.sort_values(['recall','gloss']).head(15).iloc[::-1]; fig,ax=plt.subplots(figsize=(9,6)); ax.barh(low.gloss,low.recall*100,color='#578bb3'); ax.set_xlim(0,100); ax.set_xlabel('Test recall (%)'); ax.set_title('Lowest-recall classes — only 6 test clips per class'); save('lowest_recall.png')
    lines=['# Evaluation graphs and error analysis','', 'Generated from existing experiment files; no new training or human sign review was performed.','',f'Test: {correct}/1200 correct ({100*m["accuracy"]:.2f}%), {1200-correct} incorrect. Macro-F1: {m["macro_f1"]:.4f}; top-5 accuracy: {100*m["top5"]:.2f}%. Selected checkpoint epoch: {m["checkpoint_epoch"]+1} (one-based; checkpoint stores a zero-based index).','', '## Figures','']
    for title,file in [('Training curves','training_curves.png'),('Validation comparison','validation_comparison.png'),('Full confusion matrix','confusion_matrix.png'),('Frequent confusions','top_confusions.png'),('Lowest recall','lowest_recall.png')]: lines += [f'### {title}',f'![{title}]({file})','']
    lines += ['## Most frequent errors','']+[f'- {t} predicted as {v}: {n} clips.' for n,t,v in pairs[:10]]
    lines += ['', '## Interpretation and limitations','', '- Confusion counts identify where to inspect videos; they do not establish why a sign was misclassified. Hand occlusion, timing and similar appearance are hypotheses requiring clip inspection.', '- Each class has six test examples, so one error changes class recall by 16.67 percentage points. Low-support rankings are unstable.', '- Compare variants on validation metrics. Training configurations differ; this is not a controlled estimate of the face stream alone.', '- Training accuracy was not recorded, so only recorded loss and validation curves are plotted.', '- The baseline test results were inspected before V2 development. The reused test set is not an untouched final benchmark.', '- Offline trimmed-video results do not measure webcam accuracy, unknown-sign rejection or sentence translation.', '- Two bilingual meanings remain pending. Classification scores do not validate translations.', '', '## Reproduce', '', 'Run `python evaluation/generate_graphs.py` from a checkout containing the recorded results. Requires numpy, pandas and matplotlib. Prediction-derived confusion counts were checked against the saved matrix and reported accuracy.', '', 'Prepared with automated analysis assistance. Fardin should review the figures and interpretation before presenting them.']
    (OUT/'EVALUATION_ANALYSIS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(f'Validated {len(p)} predictions: {correct} correct. Generated five figures and report.')
if __name__=='__main__': main()

