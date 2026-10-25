**English** | [한국어](../ko/03-hard-negative-mining.md)

# 3. Hard-negative mining: feed false positives back

Public datasets rarely contain the things *your* camera confuses with damage.
The fastest way to find them is to run the model on your own video and sort the results.

## Loop

1. **Detect** on your videos and save every detection:
   ```bash
   python scripts/custom_detect.py --model best.pt --source video.mp4 \
       --resolution 1280x720 --no-gnss --thresh 0.5 --sample-sec 1.5
   ```
   Each detection is saved as an annotated image (`result/`), the raw frame (`result_raw/`)
   and the predicted boxes in YOLO format (`result_label/`).
2. **Sort by hand**: move each file in `result/` into a `correct/` or `wrong/` folder.
3. **Measure precision per class**: correct ÷ saved.
4. **Add the sorted frames to the dataset** (`train` only):
   - correct → raw frame (`result_raw/`) + predicted boxes (`result_label/`) as a new labeled image
   - wrong → raw frame with an **empty label file** as a background image
   - skip every frame that came from the held-out test video
5. **Rebuild the dataset and retrain**, then repeat on the held-out video.

## Result

| | v3 | v5 (+ 127 background frames ×3) |
|---|---|---|
| Pothole detections | 138 (all 121 checked were manholes) | **0** |
| Held-out test video | 5 (all wrong) | **0** |

The false alarms also disappeared on the video the model never trained on,
so it learned "manhole ≠ pothole", not just those specific frames.

## Things to watch

- **Never import frames from the test video.** Otherwise the test score means nothing.
  Name saved frames with the video name so they can be filtered.
- **A correct frame must have all its boxes right.** If one box is wrong or one defect is missing,
  fix the label before importing, or the error is learned.
- **Zero detections is not automatically good.** If no real pothole exists in your test video,
  you cannot tell whether the model can still find one. Keep a few known positives to check recall.
- Oversample mined frames a little (I used ×3) because they are few compared to the public data.
