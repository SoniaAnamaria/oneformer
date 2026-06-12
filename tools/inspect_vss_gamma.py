"""
Inspect the vss_gamma gate across saved checkpoints, or write a copy of a
checkpoint with the gate zeroed (for the inference-time ablation).

Usage on the training VM:
  # reconstruct the gamma trajectory from periodic checkpoints
  python tools/inspect_vss_gamma.py --scan output/

  # write a zeroed-gamma copy of the final checkpoint, then eval it with
  #   python train_net.py --config-file <cfg> --eval-only MODEL.WEIGHTS output/model_final_nogamma.pth
  python tools/inspect_vss_gamma.py --zero output/model_final.pth -o output/model_final_nogamma.pth
"""
import argparse
import glob
import os
import re

import torch


def load_model_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt, ckpt.get("model", ckpt)


def gamma_keys(state, pattern="vss_gamma|refine_gamma"):
    return [k for k in state if re.search(pattern, k)]


def iteration_of(path):
    m = re.search(r"model_(\d+|final)", os.path.basename(path))
    return m.group(1) if m else path


def scan(output_dir):
    paths = sorted(
        glob.glob(os.path.join(output_dir, "model_*.pth")),
        key=lambda p: (iteration_of(p) == "final", iteration_of(p).zfill(12)),
    )
    if not paths:
        print(f"no model_*.pth found in {output_dir}")
        return
    print(f"{'iter':>10}  {'abs_mean':>9}  {'abs_max':>9}  {'std':>9}")
    for p in paths:
        _, state = load_model_state(p)
        for k in gamma_keys(state, args.match):
            g = state[k].float()
            print(
                f"{iteration_of(p):>10}  {g.abs().mean():9.4f}  "
                f"{g.abs().max():9.4f}  {g.std():9.4f}   {k}"
            )


def zero(ckpt_path, out_path):
    ckpt, state = load_model_state(ckpt_path)
    keys = gamma_keys(state, args.match)
    if not keys:
        print(f"no keys matching '{args.match}' in {ckpt_path}")
        return
    for k in keys:
        print(f"zeroing {k} (abs_mean was {state[k].abs().mean():.4f})")
        state[k] = torch.zeros_like(state[k])
    # keep only the model weights; optimizer/scheduler state is not needed for eval
    torch.save({"model": state} if "model" in ckpt else state, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", metavar="OUTPUT_DIR", help="print gamma stats per checkpoint")
    group.add_argument("--zero", metavar="CKPT", help="write a zeroed-gamma copy of CKPT")
    ap.add_argument("-o", "--out", help="output path for --zero")
    ap.add_argument(
        "--match",
        default="vss_gamma|refine_gamma",
        help="regex selecting which gamma keys to scan/zero (e.g. 'refine_gamma' only)",
    )
    args = ap.parse_args()

    if args.scan:
        scan(args.scan)
    else:
        out = args.out or args.zero.replace(".pth", "_nogamma.pth")
        zero(args.zero, out)
