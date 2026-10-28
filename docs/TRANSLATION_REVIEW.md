# Two translation conflicts requiring sign review

Official metadata repeats the gloss protibondhi for both Enemy and Handicapped.
The actual model classes include both sotru and protibondhi. Proposed textual mapping:
- sotru: Enemy / শত্রু
- protibondhi: Handicapped / প্রতিবন্ধী (source wording retained for review)

Member 2: ask a Bengali sign-language signer/teacher to confirm the actual signs, using
multiple training examples. Do not use model predictions as linguistic evidence.
Training clips:
- dataset/extracted/SignBD-Word/SignBD-Word_RGB/DATASET/sotru/p1_c_sotru.mp4
- dataset/extracted/SignBD-Word/SignBD-Word_RGB/DATASET/sotru/p4_c_sotru.mp4
- dataset/extracted/SignBD-Word/SignBD-Word_RGB/DATASET/protibondhi/p1_c_protibondhi.mp4
- dataset/extracted/SignBD-Word/SignBD-Word_RGB/DATASET/protibondhi/p4_c_protibondhi.mp4

After confirmation, record reviewer/date/evidence in manifests/translations.json and
set reviewed=true for these entries. Correct the meanings if the signer disagrees.
The other 198 entries were checked against source text; they are not claimed to have
been independently verified by a native signer. Final language review remains recommended.

RUN_REVIEW_DEMO.cmd allows interface testing now. The two unreviewed predictions display
Meaning awaiting review / অর্থ যাচাই বাকি, never their proposed translations.
RUN_DEMO.cmd remains strict and requires all labels reviewed.
