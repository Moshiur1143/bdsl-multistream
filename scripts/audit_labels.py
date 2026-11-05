"""Reproducible source-text comparison; does not verify sign meanings."""
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bilingual_labels import BilingualLabels


def normalize(value):
    return ' '.join(unicodedata.normalize('NFC', str(value)).split()).casefold()


def main():
    labels = json.loads((ROOT / 'manifests/label_map.json').read_text(encoding='utf-8-sig'))
    translations = json.loads((ROOT / 'manifests/translations.json').read_text(encoding='utf-8-sig'))
    sources = defaultdict(list)
    with (ROOT / 'manifests/official_translations_Sheet1.csv').open(encoding='utf-8-sig', newline='') as stream:
        for line, raw in enumerate(csv.DictReader(stream), 2):
            row = {key.strip(): (value or '').strip() for key, value in raw.items()}
            if row['Gloss']:
                sources[normalize(row['Gloss'])].append((line, row))
    display = BilingualLabels(ROOT / 'manifests/label_map.json', ROOT / 'manifests/translations.json', require_reviewed=False)
    rows = []
    for gloss, class_id in sorted(labels.items(), key=lambda item: item[1]):
        translation = translations[gloss]
        matches = sources[normalize(gloss)]
        issues = []
        if len(matches) != 1:
            status = 'unresolved'
            issues.append('No unique official gloss match: ' + str(len(matches)) + ' rows')
        else:
            source = matches[0][1]
            for lang, key in [('en', 'English'), ('bn', 'Bangla')]:
                if normalize(translation[lang]) != normalize(source[key]):
                    issues.append(lang + ' display differs from official text')
            status = 'needs_review' if issues else 'matches_dataset'
        if not translation.get('reviewed'):
            status = 'unresolved'
            issues.append('Existing pending meaning retained')
        rows.append(dict(class_id=class_id, gloss=gloss, english=translation['en'], bangla=translation['bn'],
                         source_rows='; '.join(str(line) for line, _ in matches),
                         source_english=' | '.join(row['English'] for _, row in matches),
                         source_bangla=' | '.join(row['Bangla'] for _, row in matches),
                         status=status, issue='; '.join(issues), reviewer='automated source comparison; human review pending',
                         review_date=datetime.now(ZoneInfo('Asia/Dhaka')).date().isoformat()))
    output = ROOT / 'docs/LABEL_REVIEW.csv'
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pending = [gloss for gloss, index in labels.items() if index in display.pending]
    for gloss in pending:
        assert display.display(labels[gloss], accepted=True)['english'] == 'Meaning awaiting review'
    assert len(labels) == 200 and set(pending) == {'sotru', 'protibondhi'}
    summary = dict(classes=len(rows), statuses=dict(Counter(row['status'] for row in rows)),
                   pending_meanings=pending, placeholder_check='passed',
                   method='Automated normalized text comparison, not linguistic or expert verification',
                   mapping_modified=False)
    (ROOT / 'results/label_audit.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
