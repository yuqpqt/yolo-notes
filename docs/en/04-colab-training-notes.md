**English** | [한국어](../ko/04-colab-training-notes.md)

# 4. Training on Colab and measuring the result

## Setup that survives disconnects

[`scripts/train_colab.py`](../../scripts/train_colab.py) saves every run to Google Drive and resumes from `last.pt`.

Zip layout: `train.py` (this script, renamed) + `dataset/{data.yaml, train/, val/}`.

```python
from google.colab import drive; drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/dataset.zip -d /content
!pip install -q ultralytics
!python train.py
```

After a disconnect, run the same cell again.

## Pitfalls I hit

| Problem | Cause | Fix |
|---|---|---|
| Resume trained on a tiny unknown dataset (`coco8`) | Once training finishes, `last.pt` stores `epoch = -1`; `resume=True` on it loses the data path | Check `torch.load(last)['epoch'] == -1` and stop instead of resuming |
| `drive.mount` fails inside the script | It needs an interactive notebook cell | Mount in a cell, have the script only check `/content/drive/MyDrive` exists |
| `data.yaml` not found | `path: .` is resolved against the working directory / Ultralytics datasets dir | Rewrite `path:` to the unzip location at start |
| `BadZipFile` | Unzipped before the upload to Drive finished | Wait for the upload; compare file size |
| Stuck at batch 0 | `cache='ram'` on a small Colab VM | Remove the `cache` option |
| `batch=64` not faster | Ultralytics accumulates gradients to a nominal batch of 64 anyway, and free Colab (2 vCPUs) is limited by data loading, not the GPU | Keep `batch=32`; ~1 min/epoch for ~4,500 images on a T4 |
| Session ends while idle | Free Colab disconnects idle sessions | Auto-resume from Drive; bump `NAME` for each new run |
| `optimizer='auto'` ignores `lr0` | `auto` picks its own learning rate | Set `optimizer='AdamW'` explicitly |

Settings I use: `epochs=100, patience=20, imgsz=640, batch=32, optimizer='AdamW', lr0=0.001`.

## Measuring on real video

Validation mAP only tells you how well the model fits *your validation set*.
My first model found almost no potholes or cracks on dashcam video, and a later one (val mAP50 0.715)
reported 138 potholes; all 121 I checked were manholes. The validation images came from the same public sources
as training, so they could not show either problem.

What I measure instead:

- **Fixed test video** that is never used for training.
- **Precision per class** at the deployment threshold (conf 0.5): correct ÷ detected.
  Target: ≥80% for pothole / crack / lane, ~50% for the rare facility classes.
- **Fast scan of long videos** with `--sample-sec 1.5`: seeking to one frame every 1.5 s
  scanned ~160 minutes of 30 fps video in ~20 minutes on a GTX 1050 Ti.
  (`--stride` decodes every frame and is slower for long videos.)

## Adding a class to a trained model without training

An older program expected 8 classes, but my model had 7.
I appended an 8th output channel to the detection head (`cv3[i][-1]` conv, zero weights and bias −20),
then set `nc=8` and added the name.
Classes 0–6 give identical results; class 7 can never fire.
It only fixes compatibility; to actually detect the class you have to train it.
