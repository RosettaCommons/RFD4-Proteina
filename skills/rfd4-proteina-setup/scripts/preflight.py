"""Check the selected Python environment and released checkpoints without sampling."""

import os
from pathlib import Path

import atomworks
import torch
import yaml

import rfproteina.conditions

repo_root = Path(__file__).resolve().parents[3]
print(f"AtomWorks: {atomworks.__file__}")
print(f"RFProteina: {rfproteina.conditions.__file__}")
if not torch.cuda.is_available():
    raise SystemExit("FAIL: CUDA is unavailable in the selected environment.")
print(f"GPU: {torch.cuda.get_device_name(0)}")
if not os.environ.get("DATA_PATH"):
    raise SystemExit("FAIL: Set DATA_PATH in .env.local.")
data = Path(os.environ["DATA_PATH"])
config = yaml.safe_load((repo_root / "configs/inference.yaml").read_text())
checkpoints = [
    data / "flow_checkpoints" / config["ckpt_name"],
    data / "ae_checkpoints" / Path(config["autoencoder_ckpt_path"]).name,
]
for path in checkpoints:
    if not path.is_file():
        raise SystemExit(f"FAIL: Missing checkpoint: {path}")
    print(f"PASS: {path}")
print("PREFLIGHT PASSED")
