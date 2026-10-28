"""Import bilingual labels from official metadata, preserving ambiguity for review."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
labels=json.loads((ROOT/'manifests/label_map.json').read_text(encoding='utf-8-sig'))
source=ROOT/'manifests/official_translations_Sheet1.csv'
rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
fix_bn={'bon':'বোন','chachi':'চাচি','nani':'নানি / দাদি','nana':'নানা / দাদা','bristi':'বৃষ্টি','chesta':'চেষ্টা','ovvash':'অভ্যাস','dhorjo':'ধৈর্য','mathabetha':'মাথাব্যথা','poriskar':'পরিষ্কার','apnader':'আপনাদের / তোমাদের'}
fix_en={'chachi':'Aunt','sharee':'Sari','subidha':'Benefit','icecream':'Ice cream','dhonnobad':'Thank you'}
numbers=['শূন্য','এক','দুই','তিন','চার','পাঁচ','ছয়','সাত','আট','নয়','দশ']
result={};changes=[]
for i,row in enumerate(rows,2):
    en=row['English'].strip();bn=row['Bangla '].strip();gloss=row['Gloss'].strip()
    if not gloss:continue
    aliases=gloss.split('/')
    if gloss=='kee':aliases=['ke'];changes.append('kee -> ke: exact class-name alignment for Who / কে')
    if en=='Enemy':aliases=['sotru'];changes.append('Official Enemy row incorrectly repeats protibondhi: proposed sotru, requires sign review')
    for key in aliases:
        key=key.strip()
        if key not in labels:continue
        if key in result:raise ValueError('Unresolved duplicate '+key)
        display_bn=numbers[int(bn)] if bn.isdigit() and int(bn)<=10 else fix_bn.get(key,bn)
        display_en=fix_en.get(key,en)
        unresolved=key in ('sotru','protibondhi')
        result[key]=dict(en=display_en,bn=display_bn,reviewed=not unresolved,
          source='SignBD-Word words_translation_gloss.xlsx; exported CSV row '+str(i),
          source_en=en,source_bn=bn,source_gloss=gloss,
          review_type='source/text checked; native signer review pending' if not unresolved else 'AMBIGUOUS source gloss; native signer verification required',
          signer_verified=False)
assert set(result)==set(labels) and len(result)==200
for key,row in result.items():
    assert row['en'] and row['bn'] and any('\u0980'<=c<='\u09ff' for c in row['bn'])
current=ROOT/'manifests/translations.json'
old=json.loads(current.read_text(encoding='utf-8-sig'))
if any(v.get('en') or v.get('bn') for v in old.values()):raise RuntimeError('Existing filled translations preserved; review changes manually')
current.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
review={'classes':200,'source_text_checked':198,'requires_sign_verification':['sotru','protibondhi'],'all_signer_verified':False,'corrections':changes,
'remaining':'Verify meanings against actual signing; do not mark these two reviewed merely to bypass the demo guard.'}
(ROOT/'results/translation_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(review,ensure_ascii=True,indent=2))
