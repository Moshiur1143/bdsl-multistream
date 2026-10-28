# Bilingual recognition output

Requirement: every supported sign class displays both English and Bengali meanings
simultaneously. Current target: all 200 SignBD-Word classes (minimum 150).

## Implementation contract
1. Preserve manifests/label_map.json used by the classifier, mapping gloss to class ID.
2. Create translations.json keyed by those exact glosses using create_translation_template.
3. Each record has en, bn, reviewed (boolean), and source fields.
4. Populate English from verified dataset translations; have a fluent Bengali reviewer
   verify Bengali word senses against the sign and dataset gloss. Do not translate
   romanized filenames blindly. Keep meaning and spelling consistent across reviewers.
5. BilingualLabels rejects missing classes, extra classes, blank text and unreviewed labels.
6. The webcam app calls display(class_id, accepted=True) only after its confidence and
   temporal-stability policy accepts a prediction. Otherwise call display() for bilingual
   no-confident-sign text. Do not emit confident labels during idle/transition frames.
7. Show English and Bangla together in a Unicode-capable UI. Use an HTML/browser UI with
   a locally bundled, appropriately licensed Bengali font and font fallback. Do not rely
   on OpenCV's built-in putText to render Bengali shaping correctly.
8. Bundle translations.json and the matching label_map.json with the trained checkpoint.
   Load local mappings at startup; no online translation API is needed during inference.

Example display ONLY (not proof that this sign is present in the dataset):
English: Thank you
বাংলা: ধন্যবাদ

## Acceptance checks before release
- All classifier classes (200 target) have reviewed English and Bengali entries.
- Every class ID resolves to the same gloss used during training.
- Test unknown IDs and rejected predictions; neither may silently select a sign.
- Visually check Bengali conjuncts and vowel marks on the actual demo PC.
- Show both labels at the same time, tied to the same accepted prediction.
- Include bilingual output in demo screenshots and presentation.

## Ownership
Member 2 prepares the complete mapping from dataset metadata and coordinates language review.
Member 3 exports the exact class mapping with the selected checkpoint.
Member 4 integrates bilingual UI and tests font rendering/rejection states.
Member 1 (project owner) checks visible labels and includes the feature in slides/demo.

## Status
Reusable label validation/display module implemented in src/bilingual_labels.py.
Actual 200-class mapping is pending dataset metadata recovery and language review.
Webcam integration, trained model and end-to-end validation are not complete.
This is bilingual isolated-word output, not continuous sentence translation.
