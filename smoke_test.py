"""
Smoke test: build OneFormer with a given config and run an inference
forward pass through it.

Usage (from repo root, with the oneformer conda env active):
    python3 smoke_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

import oneformer  # registers models and datasets
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = "configs/cityscapes/convnext/oneformer_convnext_large_bs16_90k.yaml"
TASK   = "panoptic"   # panoptic | semantic | instance


def build_cfg(config_file: str, device: str) -> object:
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_oneformer_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list([
        "MODEL.WEIGHTS",       "",
        "MODEL.DEVICE",        device,
        "MODEL.IS_TRAIN",      "False",
        "SOLVER.AMP.ENABLED",  "False",
    ])
    cfg.freeze()
    return cfg


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_image_tensor(h: int, w: int, device: str) -> torch.Tensor:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
    return torch.as_tensor(img.transpose(2, 0, 1)).to(device)


def main():
    device = pick_device()
    print(f"Device  : {device}")
    print(f"Config  : {CONFIG}")
    print(f"Task    : {TASK}")
    print()

    cfg = build_cfg(CONFIG, device)
    H = cfg.INPUT.MIN_SIZE_TEST
    W = cfg.INPUT.MAX_SIZE_TEST
    num_queries = cfg.MODEL.ONE_FORMER.NUM_OBJECT_QUERIES
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    print(f"Image   : {H}×{W}   Queries: {num_queries}   Classes: {num_classes}")
    print()

    model = build_model(cfg)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    model.eval()
    with torch.no_grad():
        inputs = [{"image": make_image_tensor(H, W, device),
                   "height": H, "width": W,
                   "task": f"The task is {TASK}"}]
        outputs = model(inputs)

    out = outputs[0]
    print(f"Output keys  : {list(out.keys())}")
    if "sem_seg" in out:
        print(f"sem_seg      : {tuple(out['sem_seg'].shape)}")
    if "panoptic_seg" in out:
        pan, segs = out["panoptic_seg"]
        print(f"panoptic_seg : {tuple(pan.shape)}, {len(segs)} segments")
    if "instances" in out:
        print(f"instances    : {len(out['instances'])} detections")
    print("\nPASSED ✓")


if __name__ == "__main__":
    main()
