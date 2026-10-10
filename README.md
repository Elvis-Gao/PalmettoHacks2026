# PoleSight

PoleSight is a PalmettoHacks 2026 prototype for visually inventorying a utility pole. It supports a custom-trained YOLO model for precise component boxes, with Gemini available as an optional fallback while a model is not yet trained.

## Run it

1. Copy `.env.example` to `.env`.
2. Put your real Gemini API key in `GEMINI_API_KEY`.
3. Run `node server.js`.
4. Open [http://localhost:3000](http://localhost:3000).

The browser app and Node gateway use no npm packages. The optional YOLO service has its own Python dependencies.

## Train and use YOLO

YOLO is integrated as the primary detector once you provide labeled data and train weights. It is designed to recognize individual pole assets, not broad Gemini-generated groups.

1. Install Python 3.10+ (Python is not installed in this workspace yet), then create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r yolo\requirements.txt
   ```

2. Add permissioned, annotated photos under `yolo/dataset`. Follow [the annotation guide](yolo/dataset/README.md) exactly; the class taxonomy is in [dataset.yaml](yolo/dataset.yaml). Every individual wire and accessory must be annotated as its own object.

3. Train a high-resolution detector for small assets and wires:

   ```powershell
   python yolo\train.py --epochs 150 --imgsz 1280
   ```

4. Start the local inference service with the resulting weights:

   ```powershell
   python yolo\service.py --weights yolo\runs\pole-components\weights\best.pt
   ```

5. Set `DETECTOR_MODE=yolo` and `YOLO_URL=http://127.0.0.1:8001` in `.env`, restart `node server.js`, and upload a photo. The status pill will say **YOLO detection complete** when your model supplied the boxes.

`DETECTOR_MODE=auto` tries YOLO first and falls back to Gemini if the local service is offline. Do not use an untrained stock YOLO model: it does not know utility-pole-specific classes.

## What it returns

For every image, the model is prompted to return JSON with:

- Visible components and normalized bounding boxes
- Visual confidence
- Non-diagnostic condition and follow-up cues
- A concise visual summary and photo limitations

## Improving detection quality

The initial prototype uses Gemini's visual object detection with a pole-specific instruction set; it is not a custom-trained detector. The app now asks for tight, non-overlapping boxes and renders long wires, guy wires, conduit, and oversized detections as compact clickable markers so they do not block other objects.

For a production-quality detector, use a two-model approach:

1. Define a stable set of classes, such as pole, crossarm, transformer, cutout, arrester, insulator, primary conductor, secondary line, telecom attachment, streetlight, and riser.
2. Collect real, permissioned field photos representing different pole types, distances, seasons, weather, lighting, and occlusion. Annotate each distinct object with a tight bounding box; keep a separate test set that is never used for training.
3. Train a dedicated object-detection model from those boxes. This project includes a self-hosted YOLO workflow that returns its boxes and confidence scores directly to the interface.
4. Keep Gemini as the optional explanation layer, not as the source of safety-critical detection. Compare the trained model against your held-out test set before relying on it in the field.

Ultralytics YOLO exposes normalized boxes, confidence, and class IDs, which the local service maps into PoleSight's overlay format. See [Ultralytics prediction results](https://docs.ultralytics.com/modes/predict/) and the [YOLO dataset format](https://docs.ultralytics.com/datasets/detect/).

## Safety and product boundaries

This is a visual-assistance demo, not an engineering assessment. It deliberately asks the model not to infer electric state, utility ownership, concealed equipment, defect severity, clearance, or safety status. A qualified field process must verify anything that affects maintenance or safety.

## Configuration

`GEMINI_MODEL` defaults to `gemini-2.5-flash`. Set it in `.env` if the provided Gemini key is provisioned for another vision-capable model.

YOLO may be used under its own licensing terms; review the current [Ultralytics licensing options](https://www.ultralytics.com/license) before a deployment beyond this hackathon prototype.
