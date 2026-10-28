# Development update: October 26

## Diagnosis from training and validation only
Face crops are present in 99.42%-99.85% of sampled training frames, and 99.65% of
validation frames. Missing crops are not the main explanation for poor face-only results.
The face-only classifier predicted only eight distinct validation classes, and its final
training loss was 4.865. This supports weak learning/generalization in the current
small, randomly initialized face CNN; it does not prove that faces contain no useful signal.
The inspected montage contains 16 signs from one training signer. Many expressions are
neutral; it is a limited visual check, not evidence about every signer.

## V2 experiment (running)
- Preserve anatomical joint identity and ordered per-joint features.
- Add already-extracted wrist-relative 3D hand shapes and finger-bone directions.
- Pool left hand, right hand and upper body separately.
- Add dilated temporal blocks and temporal attention/max pooling.
- Small synchronized temporal-crop augmentation; no sign reversal or horizontal mirroring.
- Graph V2 has 1,441,433 parameters. A 16-training-clip optimization sanity check passed;
  its loss decrease is not a recognition accuracy result.
- Train graph V2 first. Then freeze that graph and test a small residual correction from
  a frozen ImageNet-pretrained ResNet18 face encoder plus a trainable temporal head.
  This generic visual encoder is not claimed to be a facial-expression classifier.
- Compare by validation macro-F1. Retain the old baseline if V2 does not improve it.
- No new test evaluation is scheduled; the old baseline's exposed test results remain frozen.

## Bilingual output
200 entries were populated from the official dataset translation table.
198 entries were checked against source text and spelling; native signer verification is
not claimed. Source ambiguity affects sotru and protibondhi. See TRANSLATION_REVIEW.md.
Strict demo mode rejects unreviewed entries. Explicit review mode replaces the two
uncertain meanings with a bilingual pending-review message.

## Demo checks
Blank-frame API test: 35 frames, no accepted sign; bilingual responses, reset buffering,
invalid-image handling and foreign-Origin rejection checked. This is not a real camera
accuracy test. The measured latency was under concurrent training and is not an FPS claim.
Actual camera testing and confidence/unknown-sign calibration remain outstanding.

## Files
Training status: results/v2_status.json
Training log: results/v2_training.log
Launch existing baseline review demo: RUN_REVIEW_DEMO.cmd
Translation review: docs/TRANSLATION_REVIEW.md
