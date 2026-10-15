"""Run CSTrack on converted ShipsMOT sequences using trained detector weights."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shipsmot_track_entry", ROOT / "test.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    module.main()
