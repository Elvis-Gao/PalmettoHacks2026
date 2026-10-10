# PoleSight

PoleSight is a PalmettoHacks 2026 prototype for visually inventorying a utility pole. Upload a field photo and Gemini returns a structured list of visible infrastructure, including the pole, lines, equipment, and attachments, with image overlays.

## Run it

1. Copy `.env.example` to `.env`.
2. Put your real Gemini API key in `GEMINI_API_KEY`.
3. Run `node server.js`.
4. Open [http://localhost:3000](http://localhost:3000).

The app uses Node's built-in server and `fetch`, so it has no dependency install step.

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
3. Train a dedicated object-detection model from those boxes (for example, Vertex AI AutoML image object detection or a self-hosted YOLO model). Return that model's boxes and confidence scores to the interface.
4. Keep Gemini as the optional explanation layer, not as the source of safety-critical detection. Compare the trained model against your held-out test set before relying on it in the field.

Vertex AI's image object detection dataset format accepts a label plus normalized bounding-box coordinates for each photo. See the [Vertex AI dataset format](https://cloud.google.com/vertex-ai/docs/training/using-managed-datasets) and [Gemini object-detection guidance](https://ai.google.dev/gemini-api/docs/image-understanding).

## Safety and product boundaries

This is a visual-assistance demo, not an engineering assessment. It deliberately asks the model not to infer electric state, utility ownership, concealed equipment, defect severity, clearance, or safety status. A qualified field process must verify anything that affects maintenance or safety.

## Configuration

`GEMINI_MODEL` defaults to `gemini-2.5-flash`. Set it in `.env` if the provided Gemini key is provisioned for another vision-capable model.
