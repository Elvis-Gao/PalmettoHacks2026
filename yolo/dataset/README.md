# PoleSight YOLO dataset

Place the source photos and their YOLO annotations in this layout:

```text
dataset/
  images/
    train/ pole-001.jpg
    val/   pole-101.jpg
    test/  pole-201.jpg
  labels/
    train/ pole-001.txt
    val/   pole-101.txt
    test/  pole-201.txt
```

Each image must have a same-named `.txt` annotation file. Each row represents exactly one physical component:

```text
class_id x_center y_center width height
```

All four coordinates are normalized from `0` to `1`. Class IDs are defined in `../dataset.yaml`.

## Annotation rules

- One actual wire = one annotation. Do not use one box for a bundle, span, or all overhead lines.
- One physical attachment = one annotation. Do not label an entire crossarm assembly and its subcomponents as the same object.
- Keep boxes tight to the visible object. For diagonal wires, a tight axis-aligned box will include some background; do not extend it along the whole image.
- Label only objects a trained field reviewer can identify consistently. Use the same class names across every image.
- Include difficult examples: backlighting, foliage, obstructions, different camera distances, wet weather, multiple pole materials, and different attachment configurations.
- Do not train or evaluate on photos that you do not have permission to use. Keep test images from separate routes/dates from the training set.
