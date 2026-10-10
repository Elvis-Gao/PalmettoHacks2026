"""Train a custom PoleSight YOLO detector from the local labeled dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / ".ultralytics"
CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT))
MATPLOTLIB_ROOT = CONFIG_ROOT / "matplotlib"
MATPLOTLIB_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_ROOT))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PoleSight YOLO detector.")
    parser.add_argument(
        "--data",
        default=str(ROOT / "curated-pole-dataset" / "data.yaml"),
        help="YOLO dataset YAML file",
    )
    parser.add_argument(
        "--model",
        default="yolo26n-seg.pt",
        help="Base checkpoint, or a prior .pt checkpoint. The supplied Roboflow export uses segmentation polygons.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Start with a short smoke test before a longer run")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size; 640 is a safe CPU smoke-test setting")
    parser.add_argument("--batch", type=int, default=1, help="Images per batch; keep at 1 for a CPU smoke test")
    parser.add_argument("--workers", type=int, default=0, help="Data-loader workers; 0 is most stable on Windows")
    parser.add_argument("--device", default="cpu", help="cuda device ID, 'cpu', or leave empty for automatic")
    parser.add_argument("--project", default=str(ROOT / "runs"))
    parser.add_argument("--name", default="curated-smoke")
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset configuration was not found: {data_path}")

    model = YOLO(args.model)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "plots": True,
        "exist_ok": True,
    }
    if args.device:
        train_args["device"] = args.device

    model.train(**train_args)
    print(f"Training complete. Use {Path(args.project) / args.name / 'weights' / 'best.pt'} for inference.")


if __name__ == "__main__":
    main()
