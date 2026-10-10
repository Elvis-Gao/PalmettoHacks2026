"""Local HTTP adapter that exposes a custom Ultralytics YOLO model to PoleSight."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / ".ultralytics"
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT))
MATPLOTLIB_ROOT = CONFIG_ROOT / "matplotlib"
MATPLOTLIB_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_ROOT))

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO  # noqa: E402
MAX_BODY_BYTES = 16 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Dataset labels can stay specific while the app preserves its existing visual color categories.
CATEGORY_ALIASES = {
    "crossarm_brace": "hardware",
    "wire": "wire",
    "secondary_conductor": "secondary_line",
    "neutral_conductor": "neutral_line",
    "service_drop": "secondary_line",
    "telecom_cable": "telecom",
    "telecom_attachment": "telecom",
    "surge_arrester": "arrester",
    "capacitor_bank": "capacitor",
    "voltage_regulator": "regulator",
    "luminaire_arm": "streetlight",
    "guy_guard": "guy_wire",
    "guy_anchor": "guy_wire",
    "conduit_riser": "hardware",
    "ground_wire": "hardware",
    "deadend_hardware": "hardware",
    "clamp_hardware": "hardware",
}
APP_CATEGORIES = {
    "pole", "crossarm", "insulator", "primary_conductor", "secondary_line", "neutral_line",
    "transformer", "cutout_fuse", "arrester", "switch", "regulator", "capacitor", "guy_wire",
    "streetlight", "telecom", "meter", "hardware", "wire", "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained PoleSight YOLO detector locally.")
    parser.add_argument(
        "--weights",
        default=os.environ.get("YOLO_WEIGHTS", str(ROOT / "runs" / "pole-components" / "weights" / "best.pt")),
        help="Path to your trained best.pt file",
    )
    parser.add_argument("--host", default=os.environ.get("YOLO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("YOLO_PORT", "8001")))
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("YOLO_IMGSZ", "1280")))
    parser.add_argument("--max-det", type=int, default=int(os.environ.get("YOLO_MAX_DET", "300")))
    return parser.parse_args()


class Detector:
    def __init__(self, weights: Path, imgsz: int, max_det: int) -> None:
        if not weights.is_file():
            raise FileNotFoundError(
                f"Trained YOLO weights not found: {weights}. Run train.py first or pass --weights PATH."
            )
        self.model = YOLO(str(weights))
        self.weights = weights
        self.imgsz = imgsz
        self.max_det = max_det
        self.lock = threading.Lock()

    def detect(self, image: Image.Image, confidence: float, iou: float) -> dict[str, Any]:
        # Ultralytics is loaded once. A lock avoids concurrent GPU prediction conflicts in this small local service.
        with self.lock:
            result = self.model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )[0]

        detections = []
        boxes = result.boxes
        if boxes is not None:
            normalized_boxes = boxes.xyxyn.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            for index, (box, score, class_id) in enumerate(zip(normalized_boxes, confidences, classes), start=1):
                class_name = str(result.names[int(class_id)])
                category = CATEGORY_ALIASES.get(class_name, class_name if class_name in APP_CATEGORIES else "other")
                x1, y1, x2, y2 = box
                detections.append({
                    "id": f"yolo-{index}",
                    "label": class_name.replace("_", " ").title(),
                    "category": category,
                    "confidence": round(float(score), 4),
                    "display_mode": "box",
                    "bounding_box_1000": [
                        clamp_1000(y1 * 1000), clamp_1000(x1 * 1000),
                        clamp_1000(y2 * 1000), clamp_1000(x2 * 1000),
                    ],
                    "condition": "not_assessable",
                    "description": f"Custom YOLO detected a visible {class_name.replace('_', ' ')}.",
                    "action": "field_verify",
                })

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return {
            "model": self.weights.name,
            "summary": f"Custom YOLO found {len(detections)} visible pole components.",
            "detections": detections,
        }


def clamp_1000(value: float) -> int:
    return max(0, min(1000, round(value)))


def decode_image(payload: dict[str, Any]) -> Image.Image:
    if payload.get("mimeType") not in ALLOWED_MIME_TYPES:
        raise ValueError("Use a JPG, PNG, or WebP image.")
    encoded = payload.get("imageBase64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("A base64 image is required.")
    try:
        raw_image = base64.b64decode(encoded, validate=True)
        image = Image.open(BytesIO(raw_image))
        image.load()
        return image.convert("RGB")
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded image could not be decoded.") from error


def make_handler(detector: Detector) -> type[BaseHTTPRequestHandler]:
    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            self.send_json(HTTPStatus.OK, {"status": "ready", "model": detector.weights.name, "imgsz": detector.imgsz})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/detect":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            try:
                body = self.read_json_body()
                image = decode_image(body)
                confidence = bounded_float(body.get("confidence"), 0.05, 0.95, 0.25)
                iou = bounded_float(body.get("iou"), 0.05, 0.95, 0.45)
                self.send_json(HTTPStatus.OK, detector.detect(image, confidence, iou))
            except ValueError as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # Keep model internals out of the browser while leaving a useful local log.
                print(f"YOLO prediction failed: {error}")
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "YOLO prediction failed."})

        def read_json_body(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_BODY_BYTES:
                raise ValueError("Invalid or oversized request.")
            try:
                payload = json.loads(self.rfile.read(content_length))
            except json.JSONDecodeError as error:
                raise ValueError("Request body must be JSON.") from error
            if not isinstance(payload, dict):
                raise ValueError("Request body must be an object.")
            return payload

        def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"YOLO service: {format % args}")

    return RequestHandler


def bounded_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        return min(maximum, max(minimum, float(value)))
    except (TypeError, ValueError):
        return fallback


def main() -> None:
    args = parse_args()
    detector = Detector(Path(args.weights).expanduser().resolve(), args.imgsz, args.max_det)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(detector))
    print(f"YOLO detector ready at http://{args.host}:{args.port} using {detector.weights}")
    server.serve_forever()


if __name__ == "__main__":
    main()
