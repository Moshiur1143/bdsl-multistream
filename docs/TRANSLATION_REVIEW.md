# Translation review and source comparison

## Automated audit, November 5, 2026

Run `python scripts/audit_labels.py` from the repository to reproduce the audit.
The script compares all 200 model glosses and bilingual display labels with
`manifests/official_translations_Sheet1.csv`, normalizing Unicode, whitespace and case.
It writes `docs/LABEL_REVIEW.csv` and `results/label_audit.json`.

- 168 entries match the official English and Bangla text after normalization.
- 25 entries have display wording differences requiring human review.
- 7 entries have no unique exact source gloss match: apnader, apnara, chih, ke,
  shona, sotru and protibondhi. Spelling aliases or existing source-row evidence
  may explain some of these; the audit does not infer equivalence.

These are text-comparison outcomes, not judgments that all differing labels are
incorrect. Number-to-word conversion, spelling corrections and wording choices
can create differences. Consult each CSV row and the existing source metadata
before changing a label. Record the supporting evidence for any correction.

No label text, class ID or existing review flag was changed by this audit.
The existing app has 198 source-reviewed flags and two pending meanings; that
historical count must not be confused with the 168 exact matches in this audit.
The bilingual loader validated all 200 entries and confirmed that both pending
classes display the review placeholder. Visual rendering was not retested.

Prepared by automated source comparison. Rajash's personal review is pending;
no human reviewer or expert verification is claimed for this new audit.
Rajash should inspect the 32 non-matching rows, record his own decisions and
evidence, and review the remaining matches for context before signing off.

Report limitation: Labels were checked against dataset annotations; independent
sign-language expert validation was not available.

## Two known meaning conflicts

Official metadata repeats the gloss protibondhi for both Enemy and Handicapped.
The actual model classes include both sotru and protibondhi. Proposed textual mapping:
- sotru: Enemy / শত্রু
- protibondhi: Handicapped / প্রতিবন্ধী (source wording retained for review)

No sign-language expert is currently available. Keep these two meanings pending
until reliable evidence resolves the ambiguity. If a signer/teacher becomes
available, use multiple training examples. Do not use model predictions as linguistic evidence.
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
