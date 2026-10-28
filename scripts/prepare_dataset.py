"""Extract RGB + metadata and reproduce the frozen signer-disjoint protocol."""
from pathlib import Path
import sys,json,zipfile,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from download_dataset import verify

def main():
    archive=ROOT/'dataset/SignBD-Word.zip'
    if not archive.exists() or not verify(archive):raise RuntimeError('Download and verify dataset first')
    out=ROOT/'dataset/extracted';out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        items=[i for i in z.infolist() if not i.is_dir() and (i.filename.startswith('SignBD-Word/SignBD-Word_RGB/') or i.filename.startswith('SignBD-Word/Train_Test_Split/') or i.filename in ('SignBD-Word/readme.txt','SignBD-Word/words_translation_gloss.xlsx'))]
        pending=[i for i in items if not (out/i.filename).exists() or (out/i.filename).stat().st_size!=i.file_size]
        if shutil.disk_usage(out).free<sum(i.file_size for i in pending)+1024**3:raise RuntimeError('Not enough extraction space')
        for i,item in enumerate(pending):
            if not (out/item.filename).resolve().is_relative_to(out.resolve()):raise ValueError('Unsafe ZIP path')
            z.extract(item,out)
            if i%200==0:print(f'Extracted {i}/{len(pending)}',flush=True)
    base=out/'SignBD-Word';rows=[]
    for split in ('train','test'):
        file=base/f'Train_Test_Split/Class_number_200/rgb/{split}_class_200_rgb.txt'
        for line in file.read_text().splitlines():
            if not line.strip():continue
            rel,_,cid=line.strip().rsplit(maxsplit=2)
            path=base/'SignBD-Word_RGB'/rel
            signer,view,_=path.stem.split('_',2)
            assert path.is_file(),path
            rows.append(dict(path=path.relative_to(ROOT).as_posix(),relative_path=rel,gloss=path.parent.name,
                signer=signer,view=view,original_class_id=int(cid),official_split=split,
                split='test' if split=='test' else ('val' if signer in ('p15','p16') else 'train')))
    df=pd.DataFrame(rows);labels={g:i for i,g in enumerate(sorted(df.gloss.unique()))}
    assert len(df)==6000 and df.path.nunique()==6000 and len(labels)==200
    df['class_id']=df.gloss.map(labels)
    groups={s:set(df[df.split==s].signer) for s in ('train','val','test')}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    for s in groups:assert df[df.split==s].class_id.nunique()==200
    manifests=ROOT/'manifests';manifests.mkdir(exist_ok=True)
    label_file=manifests/'label_map.json'
    if label_file.exists():assert json.loads(label_file.read_text())==labels,'Existing label map mismatch'
    label_file.write_text(json.dumps(labels,ensure_ascii=False,indent=2),encoding='utf8')
    df.to_csv(manifests/'all_videos.csv',index=False)
    for s in groups:df[df.split==s].to_csv(manifests/f'{s}.csv',index=False)
    (manifests/'split_protocol.json').write_text(json.dumps({s:sorted(v) for s,v in groups.items()},indent=2))
    # Preserve official translation metadata for human review. Do not guess column meanings.
    xlsx=base/'words_translation_gloss.xlsx'
    for name,sheet in pd.read_excel(xlsx,sheet_name=None).items():
        safe=''.join(c for c in name if c.isalnum() or c in '_-') or 'sheet'
        sheet.to_csv(manifests/f'official_translations_{safe}.csv',index=False,encoding='utf-8-sig')
    sys.path.insert(0,str(ROOT/'src'))
    from bilingual_labels import create_translation_template
    target=manifests/'translations.json'
    if not target.exists():create_translation_template(label_file,target)
    print(df.groupby('split').agg(videos=('path','size'),classes=('class_id','nunique')))
    print('Manifests ready. Translation template requires language review.')
if __name__=='__main__':main()
