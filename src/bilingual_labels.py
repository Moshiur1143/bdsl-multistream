"""Validated bilingual display labels for classifier predictions (UTF-8)."""
import json
from pathlib import Path


class BilingualLabels:
    def __init__(self, label_map_path, translations_path, minimum_classes=150, require_reviewed=True):
        label_map = json.loads(Path(label_map_path).read_text(encoding='utf-8-sig'))
        translations = json.loads(Path(translations_path).read_text(encoding='utf-8-sig'))
        if not isinstance(label_map, dict) or len(label_map) < minimum_classes:
            raise ValueError(f'Expected at least {minimum_classes} model classes')
        ids = list(label_map.values())
        if any(type(i) is not int for i in ids) or sorted(ids) != list(range(len(ids))):
            raise ValueError('Model class IDs must be unique and contiguous from zero')
        if not isinstance(translations, dict) or set(translations) != set(label_map):
            raise ValueError('Translation keys must exactly match the model glosses')
        self.by_id = {}
        self.pending = set()
        for gloss, class_id in label_map.items():
            row = translations[gloss]
            if not isinstance(row, dict):
                raise ValueError(f'Invalid translation: {gloss}')
            if row.get('reviewed') is not True: self.pending.add(class_id)
            if require_reviewed and row.get('reviewed') is not True:
                raise ValueError(f'Unreviewed translation: {gloss}')
            for lang in ('en', 'bn'):
                value = row.get(lang)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'Missing {lang} translation: {gloss}')
            if not any('\u0980' <= c <= '\u09ff' for c in row['bn']):
                raise ValueError(f'Bangla script missing: {gloss}')
            self.by_id[class_id] = dict(class_id=class_id, gloss=gloss,
                                       english=row['en'].strip(), bangla=row['bn'].strip())

    def display(self, class_id=None, *, accepted=False):
        """The caller applies its calibrated confidence/temporal acceptance policy."""
        if not accepted:
            return dict(class_id=None, gloss=None, english='No confident sign detected',
                        bangla='নিশ্চিতভাবে কোনো ইশারা শনাক্ত হয়নি')
        if type(class_id) is not int or class_id not in self.by_id:
            raise ValueError('Unknown classifier class ID')
        if class_id in self.pending:
            return dict(class_id=class_id, gloss=self.by_id[class_id]['gloss'], english='Meaning awaiting review', bangla='অর্থ যাচাই বাকি')
        return dict(self.by_id[class_id])


def create_translation_template(label_map_path, output_path):
    """Create an unreviewed template; never overwrite existing translation work."""
    labels = json.loads(Path(label_map_path).read_text(encoding='utf-8-sig'))
    template = {gloss: dict(en='', bn='', reviewed=False, source='') for gloss in labels}
    with Path(output_path).open('x', encoding='utf-8') as stream:
        json.dump(template, stream, ensure_ascii=False, indent=2)

