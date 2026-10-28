# BdSL: 200-class bilingual sign recognition

Recognizes isolated Bengali sign words and displays English and Bangla labels. This is not grammatical sentence translation. Deadline: November 7, 2026.

## Current model and results

Selected model: `fusion_v2`, combining temporal hand/pose landmark graphs with face image features over 32 frames. The frozen pretrained ResNet18 face encoder is not a dedicated facial-expression classifier.

- SignBD-Word: 6,000 clips, 200 classes.
- Signer-separated train/validation/test: 4,000 / 800 / 1,200 clips.
- Validation accuracy: 86.75%; macro-F1: 0.8620.
- Test accuracy: 81.83% (982/1,200); macro-F1: 0.8132; top-5: 93.33%.

See [evaluation report](results/fusion_v2/EVALUATION_REPORT.md). Baseline test results were inspected before V2 development, so the reused test set is not an untouched final benchmark. Five signs have been tried informally on webcam; this is not a controlled live accuracy study. Unknown gestures can be misclassified and confidence filtering remains provisional.

## What GitHub contains

Source, dependencies, label mappings, split manifests, tests and recorded evaluation reports. Dataset videos, cached features, environments, trained weights and detector binaries are excluded.

**Cloning alone does not provide a runnable trained demo.** Obtain `models/fusion_v2/best.pt` and the `models/detectors/` folder from the project owner, preserving those paths. Alternatively use the prepared `BdSL_Team_Demo.zip` bundle. The dataset is not needed to run inference. Model assets only need to be copied once.

## Run the source checkout on Windows

Install Python 3.12, copy the model assets, then run from this folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.venv\Scripts\python.exe app.py --mode fusion_v2 --review-mode
```

The CPU installation is suitable for teammates without a supported NVIDIA GPU. The original training environment is recorded in `requirements.lock.txt`, including CUDA-specific PyTorch versions; do not use that lockfile for the CPU demo setup. Do not copy `.venv` between PCs. Initial installation needs internet; inference uses local assets.

Open http://127.0.0.1:5000 and allow camera access. For OBS, start Virtual Camera and select it in the camera menu. `RUN_DIRECT_CAMERA.cmd` provides the direct-camera alternative. Rotation controls correct sideways frames before inference. Keep face and both hands visible, perform one complete sign at a time and pause between signs. Frames are processed locally and not saved by the app. Keep the development server local.

The separate team ZIP includes `SETUP.cmd` and `START_DEMO.cmd`. Its CPU inference was checked on the development machine; installation on a clean teammate PC still needs verification.

## Labels and dataset

All 200 classes have bilingual mappings. Two meanings (`sotru`, `protibondhi`) remain unverified due to inconsistent official glosses. Review mode uses placeholders for pending meanings; strict mode requires completed review. Source-text review does not establish sign-language expert verification.

Dataset: [SignBD-Word on Zenodo](https://zenodo.org/records/6779843). Preserve attribution and check its license before redistributing recordings. Existing manifests and some training scripts refer to `F:\ML`; adapt paths before training elsewhere. Preserve signer splits and checkpoint preprocessing configuration.

## Project layout

- `src/`: preprocessing, models, labels and live geometry.
- `scripts/`: dataset preparation, extraction and training.
- `web/`, `app.py`, `direct_camera.py`: webcam UI and inference.
- `manifests/`: class IDs, translations and frozen splits.
- `results/`: experiments and evaluation evidence.
- `tests/`: implementation checks, not recognition accuracy evidence.
- `docs/`: team responsibilities and review notes.

## Team workflow

Each member clones the repository, creates a branch, commits their actual changes using their own identity, pushes the branch and opens a pull request. Describe what changed and how it was checked. The imported project is shared starting material, not separately completed work by each member.

The owner must grant teammates access to a private repository. Keep model and dataset backups separately from Git. Never commit credentials.
