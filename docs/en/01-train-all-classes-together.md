**English** | [한국어](../ko/01-train-all-classes-together.md)

# 1. Train all classes together, every time

## What I tried

I had a working 8-class model (potholes, cracks, lanes, ...).
Classes 2/3/4 (damaged median, delineator post, bollard) were weak, so I collected images of just those
three and fine-tuned the existing `best.pt` on them.

## What happened

Afterwards, **pothole detections on the same video dropped from 87 to 0.**
The model forgot what it already knew. This is called *catastrophic forgetting*.

## Why it happens

- **Training only moves the weights toward the data you show it.**
  If no pothole appears in the new data, nothing keeps the pothole weights in place,
  and every update is free to overwrite them.
- **Unlabeled objects are learned as background.**
  YOLO treats every pixel without a box as "not an object".
  If a pothole is visible in a bollard photo but has no pothole label, the model is told
  *this is not a pothole*. It actively learns against the old class.
- **Class ids are just positions.**
  If the new `data.yaml` has a different class order or count, id 0 in the new labels may mean
  something else than id 0 in the old model, and the head is retrained to the new meaning.

## What to do instead

1. **Keep one master dataset** containing every class, and retrain on all of it each time
   (from `yolo11n.pt` or from the previous best, but always with all data).
2. **Fix the class list and never renumber it.** Remap every new dataset to it.
3. **Label every class visible in every image**, even if the image was collected for another class.
4. **Balance instead of separating:**
   - cap large classes (~1,500–2,000 images each),
   - oversample small classes 2–5× in `train` only (originals only in `val`),
   - add ~10% background images (no boxes).
5. **Keep a fixed validation set per class**, so you can see when a class gets worse.

Keep the original downloaded datasets untouched and rebuild `train/` and `val/` from them with a script,
so the remapping and oversampling are the same every time.

## Is fine-tuning on new data ever OK?

Yes, but only if the fine-tuning data still contains the old classes (for example by mixing in
a sample of the old dataset). Training on the new classes alone almost always hurts the old ones.
