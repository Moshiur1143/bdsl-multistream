# Live food-sign investigation

Compared the same training clip p1_f_khabar.mp4 using 32 uniformly sampled frames, followed by three repeated final-frame requests, through the Flask /frame endpoint. This is a deterministic pipeline sanity check, not a live-user accuracy test or unseen evaluation.

- 640-pixel-wide JPEG input: Food score about 99.4%, but no accepted output in the four evaluated windows; intermittent hand detection blocked the stability gate.
- 960-pixel-wide JPEG input: Food ranked first and was accepted in the third and fourth evaluated windows, with the original 0.80 threshold and three-prediction stability requirement unchanged.
- Control p1_f_boi.mp4: 640 input incorrectly ranked Friend first (~90%); 960 input ranked Book first (~39%) but did not accept. Resolution does not solve every live issue.

Changes: browser requests 1280x720 ideally and sends frames bounded by 960 pixels on the longest edge; direct capture requests 1280x720 and retains up to 960 pixels instead of always shrinking to width 640. No upscaling if the source is smaller. The actual camera may not honor the requested size. Training feature code and model weights were not changed. Potential cost: higher capture and preprocessing latency.

Remaining: user must restart the direct server / reload browser and repeat live attempts. Portrait framing, motion blur, gesture timing, and source resolution remain possible causes of failures. No claim that live Food recognition is fully resolved.
