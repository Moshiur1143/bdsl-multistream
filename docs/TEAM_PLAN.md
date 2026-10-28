# Four-member work allocation
Deadline: November 7, 2026
Project root: F:\ML
Target: 200 isolated Bengali sign-word classes. The trained fusion_v2 classifier is available. These roles describe responsibilities, not completed individual contributions.

## Member 1: You — presentation and project documentation (lightest technical role)
- Maintain README: project purpose, folder contents and final run instructions.
- Assemble slides using verified results supplied by Members 2 and 3.
- Explain the problem, accessibility use case and demo in the presentation.
- Follow Member 4's demo checklist, rehearse and record a backup demo.
Deliverables: README.md, presentation/slides.pptx, presentation/demo video.
You are not responsible for model coding, CUDA setup or debugging.

## Member 2 — dataset and preprocessing
- Restore and preserve the dataset once in dataset/.
- Verify 6000 videos / 200 classes and the frozen signer-separated split.
- Own manifests, decoding, hand/pose graphs, synchronized face crops and masks.
- Validate resumable feature extraction and provide a coverage/quality report.
Deliverables: dataset manifests, preprocessing code, features and quality report.
Preserve the official test signers; do not tune preprocessing on test results.

## Member 3 — model training and experiments (heaviest technical role)
- Implement a landmark-only temporal graph baseline for all 200 classes.
- Add the face stream and fusion; compare graph-only, face-only and fusion.
- Select checkpoints on validation only; run final test evaluation after selection.
- Supply accuracy, macro-F1, per-class results, confusion matrix and limitations.
Deliverables: training code, checkpoints, configurations and results.
Never report detector success as recognition accuracy.

## Member 4 — local setup, integration and webcam demo
- Set up the Python environment and compatible GPU dependencies on the host PC.
- Integrate Member 3's checkpoint with exactly the training preprocessing.
- Implement webcam buffering, text labels, smoothing and calibrated rejection.
- Measure end-to-end latency/FPS and prepare installation/run instructions.
Deliverables: app entry point, dependency file, demo checklist and timing report.
Do not describe isolated-word classification as grammatical sentence translation.

## Shared workflow
The most powerful PC is the compute host; this does not make its owner responsible
for the difficult training work. Members 3 and 4 operate/setup it together with the owner.
Agree on the feature format, label mapping and checkpoint interface before integration.
Each member must understand the overall architecture and be able to explain their part.
Keep large datasets, features and model weights out of ordinary Git commits.
Use Git for code and maintain a separate backup of local data and deliverables.
No passwords or personal credentials belong in the project repository.

## Target schedule
Oct 25-27: local setup, persistent dataset, feature audit and baseline preparation.
Oct 28-30: baseline training, fusion and validation comparisons.
Oct 31-Nov 2: final evaluation and webcam integration.
Nov 3-5: robustness/timing checks, report, slides and backup demo.
Nov 6: freeze files and rehearse.
Nov 7: presentation.
These are target dates, not guaranteed results. Prioritize a working 200-class
baseline if fusion does not improve validation results.

Bilingual output is required for every class. See docs/BILINGUAL_OUTPUT.md for ownership and acceptance checks.
