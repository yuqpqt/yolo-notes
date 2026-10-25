**English** | [한국어](../ko/02-dataset-collection-tips.md)

# 2. Dataset collection tips (dashcam road damage)

My first model scored well on validation but found almost no potholes or cracks in my own dashcam video.
The data did not look like the camera. These are the rules I follow now.

## Choosing images

1. **Use only images from the same viewpoint as your camera.**
   For a dashcam: windshield view with sky, lanes and cars in front; damage looks small and far away.
2. **Skip** top-down photos taken by a person standing on the road, close-ups of only the damage,
   drone / satellite images, and street-view panoramas.
3. **Watch the scene type.** A dataset of only rural dirt roads or foreign highways transferred poorly to city roads.
4. **Look through the sample images on the dataset page before downloading**, and check
   the per-class counts (Roboflow *Health Check*).

## Searching for a specific class (RDD codes)

Most road-damage datasets use the codes from the Road Damage Detection (RDD) challenge.
Searching by code is faster than by name.

| RDD code | Meaning | My class id |
|---|---|---|
| D00 | Longitudinal crack | 1 Crack |
| D10 | Transverse crack | 1 Crack |
| D20 | Alligator crack | not used (labels removed) |
| D40 | Pothole | 0 Pothole |
| D43 | Faded crosswalk | — |
| D44 | Faded white line | 6 Poor_lane |

Example searches: `dashcam pothole`, `dash cam road damage`, `RDD2022`, `D44`, `white line blur`.

The same code has different names in different datasets (`D40rotation`, `pothole`, `Pothole-1`, ...).
After downloading, read the `names` list in its `data.yaml` and write the id mapping by hand.

## Export settings (Roboflow)

5. **Format:** YOLOv11 (or YOLOv8; same label format).
6. **Turn augmentation off.** Roboflow's "×3" copies are not new data, and YOLO already augments
   during training (horizontal flip, HSV, mosaic, scale, ...).
7. **Skip flipped or rotated images.** Vertical flip, 90° and 180° rotation never happen on a dashcam;
   they only teach the model scenes it will never see.
8. **Avoid `Resize: Stretch`.** Stretching a 16:9 frame to 640×640 squashes every object.
   Download the raw size, or use `Fit (black edges)`.
9. If only an augmented version exists, keep **one copy per original**.
   Roboflow file names look like `name_jpg.rf.<hash>.jpg`; the part before `.rf.` identifies the original.

## Checking labels after download

10. **Open ~20 images and look at the boxes.**
11. **Every visible defect must have a box.** An unboxed pothole is learned as "not a pothole".
12. **Label all your classes in every image**, not just the one the dataset was made for
    (e.g. a crack inside a faded-lane image).
13. **Drop datasets with sparse or tiny boxes.** A crack dataset where only a few cracks per image were
    labeled gave almost zero crack detections.
14. Remove classes you do not use (I deleted alligator-crack boxes and dropped images that had only those).
15. Label line format: `class x_center y_center width height`, all coordinates in 0–1.

## Building train / val

16. **Split by group, not by image.** Augmented copies of one original, or frames of one video,
    must stay on the same side. Otherwise val is full of near-duplicates of train and the score is inflated.
17. When extracting frames from video, take **at most one frame per second**.
18. **Background images (no boxes) ≈ 10% of train**: normal facilities (undamaged median, bollard),
    and things the model confuses with damage (manholes, crosswalks, lane markings, shadows, stains).
19. **Hold out one of your own videos completely** and use it only for testing.
