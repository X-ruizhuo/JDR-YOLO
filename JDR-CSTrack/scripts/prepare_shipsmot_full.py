"""Convert ShipsMOT sequence ZIPs to YOLO detection data and retain MOT ground truth."""

import argparse
import csv
import io
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from extract_shipsmot_half import CLASS_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def scan_archives(folder):
    archives = sorted(folder.glob("*.zip"))
    if not archives:
        raise ValueError(f"No ZIP archives found in {folder}")
    sequences = {}
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            members = defaultdict(dict)
            for info in archive.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename).parts
                if len(parts) < 3 or not parts[-3].isdigit():
                    continue
                sequence = parts[-3]
                if parts[-2] == "img1" and Path(parts[-1]).suffix.lower() in IMAGE_EXTENSIONS:
                    members[sequence].setdefault("images", []).append(info.filename)
                elif parts[-2:] == ("gt", "gt.txt"):
                    members[sequence]["gt"] = info.filename
            for sequence, data in members.items():
                if not data.get("images"):
                    continue
                if sequence in sequences:
                    raise ValueError(f"Duplicate sequence {sequence}: {sequences[sequence][0].name} and {path.name}")
                images = data["images"]
                frames = [int(Path(name).stem) for name in images]
                if len(frames) != len(set(frames)):
                    raise ValueError(f"Duplicate frame in sequence {sequence}")
                sequences[sequence] = (path, data)
    return sequences


def annotations(archive, member, sequence, frames):
    boxes = defaultdict(list)
    if not member:
        return boxes, None
    with archive.open(member) as source:
        rows = [row for row in csv.reader(io.TextIOWrapper(source, encoding="utf-8-sig")) if row]
    for row in rows:
            if len(row) < 9:
                raise ValueError(f"Sequence {sequence}: malformed GT row {row}")
    gt_frames = {int(row[0]) for row in rows}
    candidates = [offset for offset in (0, -1)
                  if all(frame + offset in frames for frame in gt_frames)]
    if not candidates:
        raise ValueError(
            f"Sequence {sequence}: GT/image frame numbering cannot be aligned by 0 or -1; "
            f"image range={min(frames)}..{max(frames)}, "
            f"GT range={min(gt_frames)}..{max(gt_frames)}")
    if len(candidates) == 2:
        if min(frames) == 0:
            offset = -1
        elif min(frames) == 1:
            offset = 0
        else:
            raise ValueError(f"Sequence {sequence}: ambiguous GT/image frame numbering")
    else:
        offset = candidates[0]
    for row in rows:
        frame = int(row[0]) + offset
        category = int(row[7])
        if not 0 <= category < len(CLASS_NAMES):
            raise ValueError(f"Sequence {sequence}: invalid class in {row}")
        if int(float(row[6])) == 1:
            boxes[frame].append((category, *map(float, row[2:6])))
    return boxes, offset


def convert(source, output, val_sequences):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    train = scan_archives(source / "train")
    test_dir = source / "test"
    test = scan_archives(test_dir) if test_dir.is_dir() else {}
    if not val_sequences <= train.keys():
        raise ValueError(f"Validation sequences missing from train ZIPs: {sorted(val_sequences - train.keys())}")
    counts = Counter()
    sequences_by_split = defaultdict(list)
    missing_gt = []
    frame_offsets = {}
    for collection, is_test in ((train, False), (test, True)):
        for sequence, (path, members) in sorted(collection.items(), key=lambda item: int(item[0])):
            split = "test" if is_test else "val" if sequence in val_sequences else "train"
            images = sorted(members["images"], key=lambda name: int(Path(name).stem))
            frames = {int(Path(name).stem) for name in images}
            if not is_test and "gt" not in members:
                raise ValueError(f"Training sequence {sequence} has no gt/gt.txt")
            if "gt" not in members:
                missing_gt.append(sequence)
            image_dir = output / "images" / split / sequence
            label_dir = output / "labels" / split / sequence
            image_dir.mkdir(parents=True, exist_ok=True)
            if "gt" in members:
                label_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                boxes, offset = annotations(archive, members.get("gt"), sequence, frames)
                frame_offsets[f"{split}/{sequence}"] = offset
                if "gt" in members:
                    gt_dir = output / "mot_gt" / split / sequence / "gt"
                    gt_dir.mkdir(parents=True, exist_ok=True)
                    with archive.open(members["gt"]) as original, (gt_dir / "gt.txt").open("wb") as target:
                        shutil.copyfileobj(original, target)
                for name in images:
                    frame = int(Path(name).stem)
                    with archive.open(name) as original:
                        image = Image.open(original)
                        width, height = image.size
                    if width <= 0 or height <= 0:
                        raise ValueError(f"Invalid image dimensions: {name}")
                    labels = []
                    for category, x, y, w, h in boxes[frame]:
                        x1, y1 = max(0.0, x), max(0.0, y)
                        x2, y2 = min(float(width), x + w), min(float(height), y + h)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        labels.append(f"{category} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} "
                                      f"{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
                    with archive.open(name) as original, (image_dir / Path(name).name).open("wb") as target:
                        shutil.copyfileobj(original, target)
                    if "gt" in members:
                        (label_dir / (Path(name).stem + ".txt")).write_text(
                            "\n".join(labels) + ("\n" if labels else ""), encoding="ascii")
                    counts[split] += 1
            sequences_by_split[split].append(sequence)
            print(f"{path.name}: sequence {sequence} -> {split}, {len(images)} images, GT offset={offset}", flush=True)

    lines = [f"train: {output / 'images/train'}", f"val: {output / 'images/val'}"]
    if test and not missing_gt:
        lines.append(f"test: {output / 'images/test'}")
    lines.append("names:")
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output / "shipsmot_full.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"images": dict(counts), "sequences": dict(sequences_by_split),
               "gt_frame_offsets": frame_offsets,
               "test_sequences_without_gt": missing_gt}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Dataset YAML: {output / 'shipsmot_full.yaml'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/downloads"),
                        help="Directory containing train/ and test/ ZIP directories")
    parser.add_argument("--output", type=Path, default=Path("/home/ShipsMOT_full"))
    parser.add_argument("--val-sequences", nargs="+", default=["11", "30", "48", "61"])
    args = parser.parse_args()
    convert(args.source.resolve(), args.output.resolve(), set(args.val_sequences))
