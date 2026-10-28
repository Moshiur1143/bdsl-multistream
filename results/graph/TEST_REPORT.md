# Graph baseline: held-out test evaluation

- Classes: 200
- Test videos: 1200 from unseen signers p5, p9 and p10.
- Correct top-1 predictions: 655/1200
- Test top-1 accuracy: 54.58%
- Test macro-F1: 53.38%
- Test top-5 accuracy: 81.83%
- Validation accuracy: 57.38%
- Selected checkpoint epoch: 47 (one-based).

The model was selected by validation macro-F1 before this test evaluation. These are trimmed-clip classification results, not live webcam accuracy. Top-5 means the correct class was among five candidates; it is not the accuracy of the single displayed word.

## Per-signer test accuracy

- p10: 49.75% (400 videos)
- p5: 62.50% (400 videos)
- p9: 51.50% (400 videos)

## Most frequent test confusions (descriptive only)

- dukkhito -> ami: 5 videos
- aangti -> sandal: 3 videos
- chesta -> doya: 3 videos
- nurse -> chosma: 3 videos
- bangladesh -> dustu: 3 videos
- kanna -> nischit: 3 videos
- sahajjo -> ghori: 3 videos
- sonibar -> aam: 3 videos
- lojja -> purush: 3 videos
- saat -> ek: 3 videos

## Remaining work

- Freeze these baseline results. Use training/validation data for any further development and disclose that the test has been inspected.
- Investigate face-stream weakness using training and validation data, without claiming fusion improved the baseline.
- Review English/Bengali labels for every class before enabling the demo.
- Evaluate live signing, idle transitions, unknown signs and end-to-end latency separately.
- Each class has only six test videos; per-class results are noisy. Signer-level generalization matters.

Checkpoint SHA-256: eb886dcc241f7e32a16b94556d7e868deb62461065f88e11b9160d6dc0e519f8
