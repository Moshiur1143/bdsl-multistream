# Selected fusion V2: evaluation and demo status

200 classes; 1,200 test clips; 982 correct top-1 predictions.

- Test top-1: 81.83%
- Test macro-F1: 81.32%
- Test top-5: 93.33%
- Validation top-1: 86.75%
- Graph V2 validation top-1: 85.25%
- Fusion validation macro-F1: 86.20%; graph V2: 84.64%.

Selected by validation macro-F1. Same held-out signer test set previously evaluated for baseline; not a fresh untouched test set.

These are trimmed video-clip results, not measured webcam accuracy. Top-5 is not the accuracy of the displayed single word. One split/seed does not establish statistical significance.

## Per-signer test accuracy

- p10: 79.50% (400 clips)
- p5: 89.00% (400 clips)
- p9: 77.00% (400 clips)

## Bilingual demo

RUN_REVIEW_DEMO.cmd now selects fusion_v2. 198 labels were checked against official source text. Two ambiguous labels remain pending; review mode displays a pending message for them. No native-signer verification is claimed for the full mapping.
RUN_DEMO.cmd is strict and will remain blocked until every translation is reviewed.
The default confidence threshold is provisional. Unknown-sign, idle/transition and real webcam testing are still needed before claiming reliable live performance.

Checkpoint SHA-256: cd391c2b86a525780caae88b4f1026e7c932cdedc6b5f74c12c5c376d414ef7c
