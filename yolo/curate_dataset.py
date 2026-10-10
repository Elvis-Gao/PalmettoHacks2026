"""Build a small, higher-quality, leakage-free dataset from a Roboflow YOLO-seg export.

This tool never alters the Roboflow export.  It picks one quality-scored image per
source-photo group, verifies its segmentation labels, splits by source group, and
creates both a reusable dataset directory and a ZIP archive.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ("crossarm", "pole", "transformer", "wire")
SPLITS = ("train", "valid", "test")
SPLIT_RATIOS = (0.80, 0.10, 0.10)


@dataclass
class Candidate:
    image: Path
    label: Path
    group: str
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    entropy: float
    pole_aspect: float
    pole_height: float
    class_counts: Counter
    score: float


def source_group(path: Path) -> str:
    """Roboflow augmentation suffixes begin with '.rf.'; remove only that suffix."""
    return path.name.split(".rf.", 1)[0]


def parse_label(path: Path) -> tuple[Counter, float, float] | None:
    counts: Counter = Counter()
    pole_aspect = 0.0
    pole_height = 0.0
    pole_area = 0.0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return None

    for line in lines:
        fields = line.split()
        if len(fields) < 7 or (len(fields) - 1) % 2:
            return None
        try:
            class_id = int(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            return None
        if class_id not in range(len(CLASS_NAMES)) or any(
            not math.isfinite(value) or value < 0 or value > 1 for value in coords
        ):
            return None
        # A zero-area polygon is not a useful segmentation annotation.
        points = np.array(coords, dtype=np.float32).reshape(-1, 2)
        if abs(cv2.contourArea(points)) < 0.000001:
            return None
        counts[class_id] += 1
        if class_id == 1:
            width = float(points[:, 0].max() - points[:, 0].min())
            height = float(points[:, 1].max() - points[:, 1].min())
            area = width * height
            if area > pole_area:
                pole_area = area
                pole_aspect = height / max(width, 0.000001)
                pole_height = height
    return (counts, pole_aspect, pole_height) if counts else None


def image_metrics(path: Path) -> tuple[int, int, float, float, float, float] | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    if min(width, height) < 320:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probability = histogram[histogram > 0] / gray.size
    entropy = float(-(probability * np.log2(probability)).sum())
    return width, height, brightness, contrast, sharpness, entropy


def quality_score(brightness: float, contrast: float, sharpness: float, entropy: float) -> float:
    # A broad, forgiving score: it rejects near-black/washed-out/blurred frames,
    # but does not force every useful cloudy or sunset image to look the same.
    exposure = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    contrast_score = min(1.0, contrast / 55.0)
    sharpness_score = min(1.0, math.log1p(sharpness) / math.log1p(400.0))
    entropy_score = min(1.0, entropy / 7.0)
    return 0.28 * exposure + 0.25 * contrast_score + 0.30 * sharpness_score + 0.17 * entropy_score


def is_useful(
    metrics: tuple[int, int, float, float, float, float],
    labels: Counter,
    pole_aspect: float,
    pole_height: float,
) -> bool:
    _, _, brightness, contrast, sharpness, entropy = metrics
    # Include a utility pole and at least one component/line.  This removes sky-only
    # and irrelevant images while keeping poles that lack a transformer.
    has_pole = labels[1] > 0
    has_component = labels[0] > 0 or labels[2] > 0 or labels[3] > 0
    return (
        has_pole
        and has_component
        # The product receives upright photos. A sideways pole or a tiny, distant
        # pole is not an ideal primary training example for that workflow.
        and pole_aspect >= 1.4
        and pole_height >= 0.25
        and 25 <= brightness <= 230
        and contrast >= 14
        and sharpness >= 8
        and entropy >= 3.5
    )


def find_candidates(source: Path) -> tuple[list[Candidate], Counter]:
    records: list[Candidate] = []
    rejected: Counter = Counter()
    for split in SPLITS:
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                rejected["missing_label"] += 1
                continue
            label_data = parse_label(label_path)
            if label_data is None:
                rejected["invalid_label"] += 1
                continue
            labels, pole_aspect, pole_height = label_data
            metrics = image_metrics(image_path)
            if metrics is None:
                rejected["invalid_or_too_small_image"] += 1
                continue
            if not is_useful(metrics, labels, pole_aspect, pole_height):
                rejected["low_quality_or_not_pole_scene"] += 1
                continue
            width, height, brightness, contrast, sharpness, entropy = metrics
            records.append(
                Candidate(
                    image=image_path,
                    label=label_path,
                    group=source_group(image_path),
                    width=width,
                    height=height,
                    brightness=brightness,
                    contrast=contrast,
                    sharpness=sharpness,
                    entropy=entropy,
                    pole_aspect=pole_aspect,
                    pole_height=pole_height,
                    class_counts=labels,
                    score=quality_score(brightness, contrast, sharpness, entropy),
                )
            )
    return records, rejected


def choose_one_per_group(candidates: list[Candidate]) -> list[Candidate]:
    best: dict[str, Candidate] = {}
    for candidate in candidates:
        current = best.get(candidate.group)
        # Favor clear frames; use annotation richness only as a tie-breaker so that
        # a noisy image is not selected just for having more wire segments.
        key = (candidate.score, sum(candidate.class_counts.values()), candidate.image.name)
        if current is None or key > (
            current.score,
            sum(current.class_counts.values()),
            current.image.name,
        ):
            best[candidate.group] = candidate
    return sorted(best.values(), key=lambda record: record.group)


def make_splits(records: list[Candidate]) -> dict[str, list[Candidate]]:
    """Find a deterministic group-level split with balanced class presence."""
    total = len(records)
    sizes = [round(total * ratio) for ratio in SPLIT_RATIOS]
    sizes[0] += total - sum(sizes)
    presence_total = Counter()
    for record in records:
        presence_total.update(record.class_counts.keys())

    best_assignment: dict[str, list[Candidate]] | None = None
    best_score = float("inf")
    for seed in range(100):
        shuffled = records[:]
        random.Random(seed).shuffle(shuffled)
        assignment = {split: [] for split in SPLITS}
        position = 0
        for split, size in zip(SPLITS, sizes):
            assignment[split] = shuffled[position : position + size]
            position += size
        error = 0.0
        for split, ratio in zip(SPLITS, SPLIT_RATIOS):
            present = Counter()
            for record in assignment[split]:
                present.update(record.class_counts.keys())
            for class_id in range(len(CLASS_NAMES)):
                target = max(1, presence_total[class_id] * ratio)
                error += ((present[class_id] - target) / target) ** 2
        if error < best_score:
            best_score, best_assignment = error, assignment
    assert best_assignment is not None
    return best_assignment


def write_dataset(output: Path, assignment: dict[str, list[Candidate]], report: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for split in SPLITS:
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)
        for record in assignment[split]:
            shutil.copy2(record.image, output / split / "images" / record.image.name)
            shutil.copy2(record.label, output / split / "labels" / record.label.name)

    (output / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
        "nc: 4\nnames: ['crossarm', 'pole', 'transformer', 'wire']\n",
        encoding="utf-8",
    )
    shutil.copy2(report, output / "curation-report.csv")
    (output / "README.txt").write_text(
        "Curated from the original Roboflow export. One quality-scored, upright image with "
        "a substantial pole was retained per source-photo group; source groups are kept "
        "wholly separate across the splits.\n"
        "Classes: crossarm, pole, transformer, wire. This is a four-class baseline; it does "
        "not label individual accessory types or individual conductor roles.\n",
        encoding="utf-8",
    )


def write_report(path: Path, selected: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "source_group",
                "file",
                "width",
                "height",
                "quality_score",
                "brightness",
                "contrast",
                "sharpness",
                "entropy",
                "pole_aspect",
                "pole_height",
                *CLASS_NAMES,
            ]
        )
        for record in selected:
            writer.writerow(
                [
                    "",
                    record.group,
                    record.image.name,
                    record.width,
                    record.height,
                    f"{record.score:.4f}",
                    f"{record.brightness:.2f}",
                    f"{record.contrast:.2f}",
                    f"{record.sharpness:.2f}",
                    f"{record.entropy:.2f}",
                    f"{record.pole_aspect:.3f}",
                    f"{record.pole_height:.3f}",
                    *(record.class_counts[class_id] for class_id in range(len(CLASS_NAMES))),
                ]
            )


def write_zip(dataset: Path, archive: Path) -> None:
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(dataset.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(dataset.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("yolo/roboflow-export"))
    parser.add_argument("--output", type=Path, default=Path("yolo/curated-pole-dataset"))
    parser.add_argument("--archive", type=Path, default=Path("yolo/pole-detection-curated.zip"))
    args = parser.parse_args()

    candidates, rejected = find_candidates(args.source)
    selected = choose_one_per_group(candidates)
    if not selected:
        raise SystemExit("No suitable images found; no dataset was created.")
    report = args.output.parent / "curation-report.csv"
    write_report(report, selected)
    assignment = make_splits(selected)
    write_dataset(args.output, assignment, report)
    write_zip(args.output, args.archive)

    print(f"Scanned usable variants: {len(candidates)}")
    print(f"Rejected: {dict(rejected)}")
    print(f"Selected source-photo groups: {len(selected)}")
    for split in SPLITS:
        counts = Counter()
        for record in assignment[split]:
            counts.update(record.class_counts)
        print(f"{split}: {len(assignment[split])} images, instances: " + ", ".join(
            f"{CLASS_NAMES[class_id]}={counts[class_id]}" for class_id in range(len(CLASS_NAMES))
        ))
    print(f"Dataset: {args.output.resolve()}")
    print(f"ZIP: {args.archive.resolve()}")


if __name__ == "__main__":
    main()
