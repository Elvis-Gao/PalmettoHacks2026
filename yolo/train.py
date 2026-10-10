"""Train a custom PoleSight YOLO detector from the local labeled dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PoleSight YOLO detector.")
    parser.add_argument("--data", default=str(ROOT / "dataset.yaml"), help="YOLO dataset YAML file")
    parser.add_argument("--model", default="yolo26n.pt", help="Base checkpoint, or a prior .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1280, help="Use high resolution for small accessories and wires")
    parser.add_argument("--batch", type=int, default=-1, help="-1 lets Ultralytics choose a safe batch size")
    parser.add_argument("--device", default=None, help="cuda device ID, 'cpu', or leave empty for automatic")
    parser.add_argument("--project", default=str(ROOT / "runs"))
    parser.add_argument("--name", default="pole-components")
    parser.add_argument("--patience", type=int, default=35)
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
