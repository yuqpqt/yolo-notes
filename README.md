**English** | [한국어](README.ko.md)

# yolo-notes — Lessons from training a road-damage detector with YOLO11

Notes and scripts from training a YOLO11n model to detect road damage
(potholes, cracks, faded lane paint, damaged road facilities) on **dashcam video** from Korean roads.

These are things I did not know when I started and learned the hard way.
Each lesson comes with the experiment that taught it.

## Key lessons

1. **Train all classes together, every time.**
   Fine-tuning a working model on only the new classes made it forget the old ones
   (pothole detections: 87 → 0). Keep one master dataset and retrain on all of it.
   → [docs/en/01-train-all-classes-together.md](docs/en/01-train-all-classes-together.md)
2. **Your training images must look like your deployment camera.**
   Top-down close-ups, drone shots and squashed (stretched) images did not transfer to dashcam footage.
   → [docs/en/02-dataset-collection-tips.md](docs/en/02-dataset-collection-tips.md)
3. **Turn off augmentation when exporting from Roboflow, and skip flipped/rotated images.**
   YOLO augments during training; a dashcam never sees the road upside down.
4. **Feed false positives back as background images.**
   Adding frames where the model mistook manholes for potholes removed those false alarms
   (pothole false positives: 138 → 0).
   → [docs/en/03-hard-negative-mining.md](docs/en/03-hard-negative-mining.md)
5. **Validation mAP is not the goal — measure on your own video.**
   One model with val mAP50 0.715 detected 138 "potholes" on my dashcam video; all 121 I checked were manholes.
   Keep one video out of training as a fixed test.
   → [docs/en/04-colab-training-notes.md](docs/en/04-colab-training-notes.md) (also covers Colab pitfalls)

## Classes

| id | Class | Notes |
|---|---|---|
| 0 | Pothole | |
| 1 | Crack | longitudinal + transverse merged (alligator cracks excluded) |
| 2 | Damaged_median | |
| 3 | Damaged_guide post | delineator post |
| 4 | Damaged_bollard | |
| 5 | Damaged_traffic sign | no data, not a target |
| 6 | Poor_lane | faded lane paint |

## Results so far

Detection count on 4 dashcam videos (~160 min, one frame every 1.5 s, conf 0.5):

| Class | v3 | v5 (+ false-positive backgrounds) |
|---|---|---|
| Pothole | 138 (all 121 checked were manholes) | 0 |
| Crack | 609 | 226 |
| Poor_lane | 132 | 67 |

Validation (v5, best epoch): precision 0.735, recall 0.695, mAP50 0.734, mAP50-95 0.468.
Precision for crack and lane on real video is still being measured.

## Scripts

| File | What it does |
|---|---|
| [scripts/train_colab.py](scripts/train_colab.py) | Colab training that saves to Google Drive and resumes automatically after a disconnect |
| [scripts/custom_detect.py](scripts/custom_detect.py) | Runs a model on images / video / camera and saves the annotated frame, raw frame and YOLO label for every detection |

Script comments are in Korean.

### custom_detect.py usage

```bash
python custom_detect.py --model best.pt --source video.mp4 --resolution 1280x720 --no-gnss --thresh 0.5
```

| Option | Description |
|---|---|
| `--model` | Path to the YOLO model (required) |
| `--source` | Image file, folder, video file, `usb0` (USB camera) or `picamera0` (required) |
| `--thresh` | Minimum confidence to draw and save (default 0.5) |
| `--resolution` | Display / save resolution, `WxH` |
| `--device` | `cuda` (default) or `cpu` |
| `--record` | Record the annotated video to `demo1.avi` |
| `--stride N` | Video: run detection on every N-th frame |
| `--sample-sec S` | Video: jump to one frame every S seconds (fast scan of long videos) |
| `--exclude` | Class names or ids to ignore, e.g. `"Poor_lane,6"` |
| `--gnss-port`, `--gnss-baud`, `--no-gnss` | GNSS (NMEA) serial input for location overlay |

Keys: `q` quit, `s` pause, `p` save a screenshot.
Detections are saved at most once per second to `result/` (with boxes), `result_raw/` (original) and
`result_label/` (YOLO txt), named `{class}_{video}_f{frame}`.

## Not included

Dashcam videos and frames (license plates, faces, location), third-party dataset images
(see each Roboflow page for its license) and model weights.

## License

[MIT](LICENSE)
