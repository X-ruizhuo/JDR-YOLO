"""Extract a sequence-balanced ShipsMOT subset directly from the train ZIP files."""

import argparse
import csv
import io
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image


CLASS_NAMES = (
    "Container ship", "General cargo ship", "Bulk cargo carrier", "Ferry boat",
    "Tug", "Kayak", "Oil carrier", "Ore carrier", "Fishing boat",
    "Passenger ship", "Sailing ship", "Barge", "War ship", "Canoe",
    "Speed boat",
)
VAL_SEQUENCES = {"11", "30", "48", "61"}
ARCHIVES = (
    ("train.zip", lambda seq: True),
    ("train (2).zip", lambda seq: seq != "43"),
    ("train (3).zip", lambda seq: seq == "70"),
)


def selected_sequences(archive, include):
    names = archive.namelist()
    sequences = sorted(
        {name.split("/")[0] for name in names
         if name.endswith("/gt/gt.txt") and include(name.split("/")[0])},
        key=int,
    )
    return names, sequences


def read_annotations(archive, sequence):
    annotations = defaultdict(list)
    with archive.open("{}/gt/gt.txt".format(sequence)) as source:
        for row in csv.reader(io.TextIOWrapper(source, encoding="utf-8-sig")):
            if not row:
                continue
            if len(row) < 9:
                raise ValueError("Sequence {}: invalid MOT row {}".format(sequence, row))
            frame = int(row[0]) - 1  # MOT frames start at 1; img1 files start at 0.
            category = int(row[7])
            if frame < 0 or not 0 <= category < len(CLASS_NAMES):
                raise ValueError("Sequence {}: invalid frame or class {}".format(sequence, row))
            if int(float(row[6])) == 1:
                annotations[frame].append((category, *map(float, row[2:6])))
    return annotations


def write_yaml(output):
    lines = [
        "train: {}/images/train".format(output.as_posix()),
        "val: {}/images/val".format(output.as_posix()),
        "names:",
    ]
    lines += ["  {}: {}".format(index, name) for index, name in enumerate(CLASS_NAMES)]
    (output / "shipsmot_half.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract(source, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory is not empty: {}".format(output))

    # Check the intended subset before writing gigabytes of images.
    sequence_count = 0
    image_count = 0
    train_classes = set()
    for archive_name, include in ARCHIVES:
        path = source / archive_name
        if not path.is_file():
            raise FileNotFoundError(path)
        with zipfile.ZipFile(path) as archive:
            names, sequences = selected_sequences(archive, include)
            sequence_count += len(sequences)
            for sequence in sequences:
                images = [name for name in names if name.startswith("{}/img1/".format(sequence))
                          and name.lower().endswith((".jpg", ".jpeg", ".png"))]
                if not images:
                    raise ValueError("Sequence {} has no images".format(sequence))
                image_count += len(images)
                if sequence not in VAL_SEQUENCES:
                    train_classes.update(cat for boxes in read_annotations(archive, sequence).values()
                                         for cat, *_ in boxes)
    if sequence_count != 40 or image_count != 18812 or train_classes != set(range(15)):
        raise ValueError(
            "Unexpected archive version: {} sequences, {} images, training classes {}".format(
                sequence_count, image_count, sorted(train_classes)
            )
        )
    print("Checked {} sequences and {} images; extracting to {}".format(
        sequence_count, image_count, output), flush=True)

    counts = defaultdict(int)
    for archive_name, include in ARCHIVES:
        with zipfile.ZipFile(source / archive_name) as archive:
            names, sequences = selected_sequences(archive, include)
            for sequence in sequences:
                split = "val" if sequence in VAL_SEQUENCES else "train"
                annotations = read_annotations(archive, sequence)
                images = sorted(name for name in names if name.startswith("{}/img1/".format(sequence))
                                and name.lower().endswith((".jpg", ".jpeg", ".png")))
                image_frames = {int(Path(name).stem) for name in images}
                if set(annotations) - image_frames:
                    raise ValueError("Sequence {}: annotations refer to missing frames".format(sequence))

                image_dir = output / "images" / split / sequence
                label_dir = output / "labels" / split / sequence
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                for name in images:
                    filename = Path(name).name
                    frame = int(Path(name).stem)
                    with archive.open(name) as data:
                        with Image.open(data) as image:
                            width, height = image.size
                    labels = []
                    for category, x, y, box_width, box_height in annotations[frame]:
                        x1, y1 = max(0.0, x), max(0.0, y)
                        x2 = min(float(width), x + box_width)
                        y2 = min(float(height), y + box_height)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        labels.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                            category, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                            (x2 - x1) / width, (y2 - y1) / height,
                        ))
                    with archive.open(name) as data, (image_dir / filename).open("wb") as target:
                        shutil.copyfileobj(data, target)
                    (label_dir / "{}.txt".format(Path(filename).stem)).write_text(
                        "\n".join(labels) + ("\n" if labels else ""), encoding="ascii"
                    )
                    counts[split] += 1
                print("{}: sequence {} -> {}, {} images".format(
                    archive_name, sequence, split, len(images)), flush=True)

    write_yaml(output)
    print("Done: train={}, val={}, total={}".format(
        counts["train"], counts["val"], sum(counts.values())))
    print("Data config: {}".format(output / "shipsmot_half.yaml"))


if __name__ == "__main__":
    if sys.version_info < (3, 8):
        raise SystemExit("This project requires Python 3.8 or newer; activate the shipsmot environment.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/downloads/train"),
                        help="Directory containing the original train ZIP files")
    parser.add_argument("--output", type=Path, default=Path("/home/ShipsMOT_half"),
                        help="New directory for YOLO images, labels, and YAML")
    args = parser.parse_args()
    extract(args.source.resolve(), args.output.resolve())
