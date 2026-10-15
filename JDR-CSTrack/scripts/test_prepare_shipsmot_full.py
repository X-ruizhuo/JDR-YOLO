import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from prepare_shipsmot_full import annotations, convert


def make_zip(path, sequence, image_frames, gt_frames=None):
    image = io.BytesIO()
    Image.new("RGB", (100, 100)).save(image, format="JPEG")
    with zipfile.ZipFile(path, "w") as archive:
        for frame in image_frames:
            archive.writestr(f"{sequence}/img1/{frame:06}.jpg", image.getvalue())
        if gt_frames is not None:
            rows = [f"{frame},1,10,20,30,40,1,9,1" for frame in gt_frames]
            archive.writestr(f"{sequence}/gt/gt.txt", "\n".join(rows))


class PrepareShipsMOTFullTest(unittest.TestCase):
    def test_zero_and_one_based_image_names(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            (source / "train").mkdir(parents=True)
            make_zip(source / "train" / "train.zip", "11", [0, 1], [1, 2])
            make_zip(source / "train" / "train (4).zip", "113", [1, 2], [1, 2])
            output = Path(temp) / "output"
            convert(source, output, {"11"})
            self.assertTrue((output / "labels/val/11/000000.txt").read_text())
            self.assertTrue((output / "labels/train/113/000001.txt").read_text())

    def test_unmatched_gt_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.zip"
            make_zip(path, "113", [1, 2], [1, 4])
            with zipfile.ZipFile(path) as archive:
                with self.assertRaisesRegex(ValueError, "frame numbering"):
                    annotations(archive, "113/gt/gt.txt", "113", {1, 2})


if __name__ == "__main__":
    unittest.main()
