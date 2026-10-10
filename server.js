import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';

await loadEnvFile();

const PORT = Number(process.env.PORT || 3000);
const MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const DETECTOR_MODE = (process.env.DETECTOR_MODE || 'auto').toLowerCase();
const YOLO_URL = process.env.YOLO_URL?.replace(/\/+$/, '');
const YOLO_CONFIDENCE = clampNumber(process.env.YOLO_CONFIDENCE, 0.1, 0.95, 0.25);
const YOLO_IOU = clampNumber(process.env.YOLO_IOU, 0.1, 0.95, 0.45);
const PUBLIC_DIRECTORY = join(process.cwd(), 'public');
const MAX_BODY_BYTES = 16 * 1024 * 1024;

const MIME_TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml'
};

const INSPECTION_PROMPT = `You are PoleSight, a careful visual inspection assistant for utility distribution poles.

Analyze this ONE photograph. Identify only components that are visibly supported by the image; never infer hidden assets, energized state, ownership, or an engineering defect. The pole could belong to Dominion Energy, but do not claim a utility owner from visual appearance alone.

Look for: the pole structure; crossarms; insulators; primary conductors; secondary/service wires; neutral/communications lines; transformers; cutouts/fuses; arresters; switches; regulators; capacitors; guy wires; streetlights; meters; telecom/cable attachments; labels; braces; conduit and other hardware. Use "other" only where no standard category fits.

Return valid JSON only, without Markdown or comments, using this exact top-level shape:
{
  "summary": "one concise, plain-language visual summary",
  "overall_condition": "appears_normal|needs_review|not_assessable",
  "safety_notes": ["short non-diagnostic observation or photo limitation"],
  "detections": [
    {
      "id": "d1",
      "label": "human-readable component name",
      "category": "pole|crossarm|insulator|primary_conductor|secondary_line|neutral_line|transformer|cutout_fuse|arrester|switch|regulator|capacitor|guy_wire|streetlight|telecom|meter|hardware|other",
      "confidence": 0.0,
      "display_mode": "box|marker",
      "bounding_box_1000": [ymin, xmin, ymax, xmax],
      "anchor_point_1000": [y, x],
      "condition": "appears_normal|needs_review|not_assessable",
      "description": "what is visibly present, concise",
      "action": "none|review_photo|field_verify"
    }
  ]
}

Localization rules are important:
1. Detect distinct physical assets only. Do not return a group or assembly box that contains several independently visible components. For example, identify a crossarm and individual insulators only when they are visibly separable; never add an extra "crossarm assembly" box around them.
2. Use a compact box only for one compact physical object. The box must be tight to the visible object and must not cover unrelated sky, landscape, wires, or nearby equipment.
3. Use "marker" (not "box") for conductors, service/telecom wires, guy wires, conduit, or any tall/wide object whose tight box would cross much of the image or hide other detections. For a marker, provide an anchor_point_1000 on a clear visible section. For a box, anchor_point_1000 may be omitted.
4. Avoid duplicate or heavily overlapping detections. Never return a full-image box. Prefer 12 or fewer high-confidence, useful detections. Omit uncertain components rather than guessing.

Bounding boxes use [ymin, xmin, ymax, xmax] normalized to 0–1000. Anchor points use [y, x] normalized to 0–1000. Use only values in range, and do not include detections you cannot locate. Confidence is your visual confidence, from 0 to 1. This result is decision support, not an electrical safety clearance or engineering inspection.`;

createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);

    if (request.method === 'POST' && url.pathname === '/api/analyze') {
      await analyzeImage(request, response);
      return;
    }

    if (request.method === 'GET') {
      await serveStatic(url.pathname, response);
      return;
    }

    respondJson(response, 405, { error: 'Method not allowed.' });
  } catch (error) {
    console.error(error);
    respondJson(response, 500, { error: 'Something went wrong while serving the request.' });
  }
}).listen(PORT, () => {
  console.log(`PoleSight is ready at http://localhost:${PORT}`);
  console.log(`Detector mode: ${DETECTOR_MODE}${YOLO_URL ? ` / YOLO: ${YOLO_URL}` : ''}`);
  console.log(`Gemini fallback model: ${MODEL}`);
});

async function analyzeImage(request, response) {
  const body = await readJsonBody(request);
  const { imageBase64, mimeType } = body || {};

  if (typeof imageBase64 !== 'string' || !imageBase64.length || typeof mimeType !== 'string') {
    respondJson(response, 400, { error: 'Please select a valid image file.' });
    return;
  }

  if (!['image/jpeg', 'image/png', 'image/webp'].includes(mimeType)) {
    respondJson(response, 400, { error: 'Use a JPG, PNG, or WebP image.' });
    return;
  }

  if (DETECTOR_MODE !== 'gemini' && YOLO_URL) {
    try {
      const analysis = await analyzeWithYolo(imageBase64, mimeType);
      respondJson(response, 200, { analysis, provider: 'yolo' });
      return;
    } catch (error) {
      console.warn('YOLO detection unavailable:', error.message);
      if (DETECTOR_MODE === 'yolo') {
        respondJson(response, 503, { error: `YOLO detector is unavailable: ${error.message}` });
        return;
      }
    }
  }

  if (DETECTOR_MODE === 'yolo') {
    respondJson(response, 503, {
      error: 'YOLO mode is selected but YOLO_URL is not configured. Start the local YOLO service and set YOLO_URL in .env.'
    });
    return;
  }

  if (!process.env.GEMINI_API_KEY) {
    respondJson(response, 503, {
      error: 'No detector is ready. Configure YOLO_URL for your trained YOLO model, or add GEMINI_API_KEY for the Gemini fallback.'
    });
    return;
  }

  await analyzeWithGemini(imageBase64, mimeType, response);
}

async function analyzeWithYolo(imageBase64, mimeType) {
  const yoloResponse = await fetch(`${YOLO_URL}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(20_000),
    body: JSON.stringify({
      imageBase64,
      mimeType,
      confidence: YOLO_CONFIDENCE,
      iou: YOLO_IOU
    })
  });
  const payload = await yoloResponse.json().catch(() => ({}));
  if (!yoloResponse.ok) throw new Error(payload.error || `YOLO service returned ${yoloResponse.status}.`);

  const detections = Array.isArray(payload.detections) ? payload.detections : [];
  return normalizeAnalysis({
    summary: payload.summary || `Custom YOLO model found ${detections.length} visible components.`,
    overall_condition: 'not_assessable',
    safety_notes: ['YOLO detections identify visible components only; field verification is required.'],
    detections: detections.map((detection) => ({ ...detection, source: 'yolo' }))
  });
}

async function analyzeWithGemini(imageBase64, mimeType, response) {
  const apiResponse = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(MODEL)}:generateContent`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': process.env.GEMINI_API_KEY
      },
      body: JSON.stringify({
        contents: [{
          role: 'user',
          parts: [
            { text: INSPECTION_PROMPT },
            { inlineData: { mimeType, data: imageBase64 } }
          ]
        }],
        generationConfig: {
          temperature: 0.1,
          responseMimeType: 'application/json'
        }
      })
    }
  );

  const payload = await apiResponse.json().catch(() => ({}));
  if (!apiResponse.ok) {
    const detail = payload?.error?.message || 'Gemini could not analyze this image.';
    console.error('Gemini request failed:', apiResponse.status, detail);
    respondJson(response, apiResponse.status >= 400 && apiResponse.status < 500 ? 502 : 503, {
      error: `Gemini request failed: ${detail}`
    });
    return;
  }

  const text = payload?.candidates?.[0]?.content?.parts
    ?.map((part) => part.text || '')
    .join('')
    .trim();

  const analysis = parseAnalysis(text);
  if (!analysis) {
    console.error('Could not parse Gemini response:', text?.slice(0, 500));
    respondJson(response, 502, { error: 'Gemini returned an unreadable inspection. Please try the image again.' });
    return;
  }

  respondJson(response, 200, { analysis: normalizeAnalysis(analysis), provider: 'gemini' });
}

function parseAnalysis(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) return null;
    try { return JSON.parse(match[0]); } catch { return null; }
  }
}

function normalizeAnalysis(analysis) {
  const validConditions = new Set(['appears_normal', 'needs_review', 'not_assessable']);
  const validActions = new Set(['none', 'review_photo', 'field_verify']);
  const detections = Array.isArray(analysis.detections) ? analysis.detections : [];

  return {
    summary: typeof analysis.summary === 'string' ? analysis.summary : 'No visual summary was returned.',
    overall_condition: validConditions.has(analysis.overall_condition) ? analysis.overall_condition : 'not_assessable',
    safety_notes: Array.isArray(analysis.safety_notes)
      ? analysis.safety_notes.filter((note) => typeof note === 'string').slice(0, 5)
      : [],
    detections: detections
      .filter((detection) => detection && typeof detection === 'object')
      .map((detection, index) => normalizeDetection(detection, index, validConditions, validActions))
  };
}

function normalizeDetection(detection, index, validConditions, validActions) {
  const category = String(detection.category || 'other');
  const box = normalizeBox(detection.bounding_box_1000);
  const requestedMode = detection.display_mode === 'marker' ? 'marker' : 'box';
  const displayMode = requestedMode === 'marker' || (detection.source !== 'yolo' && shouldUseMarker(category, box)) ? 'marker' : 'box';
  const anchor = normalizePoint(detection.anchor_point_1000) || centerOf(box);

  return {
    id: String(detection.id || `d${index + 1}`),
    label: String(detection.label || 'Unlabeled component').slice(0, 80),
    category,
    confidence: Math.max(0, Math.min(1, Number(detection.confidence) || 0)),
    display_mode: displayMode,
    bounding_box_1000: box,
    anchor_point_1000: anchor,
    condition: validConditions.has(detection.condition) ? detection.condition : 'not_assessable',
    description: typeof detection.description === 'string' ? detection.description : 'Visible component.',
    action: validActions.has(detection.action) ? detection.action : 'review_photo'
  };
}

function clampNumber(value, min, max, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(max, Math.max(min, number)) : fallback;
}

function shouldUseMarker(category, box) {
  if (!box) return true;
  const [y1, x1, y2, x2] = box;
  const width = x2 - x1;
  const height = y2 - y1;
  const area = (width * height) / 1_000_000;
  const linearCategories = new Set(['primary_conductor', 'secondary_line', 'neutral_line', 'guy_wire', 'telecom']);

  // These shapes routinely overlap the entire image, obscuring smaller assets beneath them.
  if (linearCategories.has(category) && (width > 180 || height > 180)) return true;
  if (category !== 'pole' && (area > 0.28 || width > 900 || height > 900)) return true;
  return false;
}

function normalizeBox(box) {
  if (!Array.isArray(box) || box.length !== 4) return null;
  const values = box.map((value) => Math.max(0, Math.min(1000, Number(value) || 0)));
  const [y1, x1, y2, x2] = values;
  return y2 > y1 && x2 > x1 ? values : null;
}

function normalizePoint(point) {
  if (!Array.isArray(point) || point.length !== 2) return null;
  return point.map((value) => Math.max(0, Math.min(1000, Number(value) || 0)));
}

function centerOf(box) {
  if (!box) return [500, 500];
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

async function serveStatic(pathname, response) {
  const requestedPath = pathname === '/' ? 'index.html' : pathname;
  const safePath = normalize(requestedPath).replace(/^[/\\]+/, '').replace(/^(\.\.[\\/])+/, '');
  const filePath = join(PUBLIC_DIRECTORY, safePath);

  if (!filePath.startsWith(PUBLIC_DIRECTORY) || !existsSync(filePath)) {
    respondJson(response, 404, { error: 'Not found.' });
    return;
  }

  const fileInfo = await stat(filePath);
  if (!fileInfo.isFile()) {
    respondJson(response, 404, { error: 'Not found.' });
    return;
  }

  const file = await readFile(filePath);
  response.writeHead(200, {
    'Content-Type': MIME_TYPES[extname(filePath)] || 'application/octet-stream',
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff'
  });
  response.end(file);
}

function readJsonBody(request) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    request.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error('Image is too large. Choose an image below 10 MB.'));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => {
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
      catch { reject(new Error('Invalid request body.')); }
    });
    request.on('error', reject);
  });
}

function respondJson(response, status, value) {
  if (response.headersSent) return;
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  response.end(JSON.stringify(value));
}

async function loadEnvFile() {
  const envFile = join(process.cwd(), '.env');
  if (!existsSync(envFile)) return;

  // Small .env loader so this demo has zero package dependencies.
  // Existing environment variables intentionally take precedence.
  const contents = await readFile(envFile, 'utf8');
  for (const line of contents.split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$/);
    if (!match || process.env[match[1]] !== undefined) continue;
    process.env[match[1]] = match[2].replace(/^['"]|['"]$/g, '');
  }
}
