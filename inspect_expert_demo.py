from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tgod_sd.utils import ensure_dir, load_expert_demo
from tgod_sd.visualization.plotting import plot_expert_xyz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_demo", type=str, default="data/expert_demo.npy")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    demo = load_expert_demo(args.expert_demo)

    print(f"shape: {demo.shape}")
    names = ["x", "y", "z", "rx", "ry", "rz", "vx", "vy", "vz", "wx", "wy", "wz"]
    for i, name in enumerate(names):
        col = demo[:, i]
        print(f"{i:02d} {name:>2s}: min={col.min(): .6f}, max={col.max(): .6f}, mean={col.mean(): .6f}, std={col.std(): .6f}")

    plot_path = Path(out) / "expert_tcp_xyz.png"
    plot_expert_xyz(demo, plot_path)
    print(f"saved: {plot_path}")


if __name__ == "__main__":
    main()
