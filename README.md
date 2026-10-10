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

## Safety and product boundaries

This is a visual-assistance demo, not an engineering assessment. It deliberately asks the model not to infer electric state, utility ownership, concealed equipment, defect severity, clearance, or safety status. A qualified field process must verify anything that affects maintenance or safety.

## Configuration

`GEMINI_MODEL` defaults to `gemini-2.5-flash`. Set it in `.env` if the provided Gemini key is provisioned for another vision-capable model.
