from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_expert_xyz(expert: np.ndarray, output_path: str | Path) -> None:
    expert = np.asarray(expert, dtype=np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(expert[:, 0], expert[:, 1], expert[:, 2], label="expert")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_expert_vs_generated(expert: np.ndarray, generated: np.ndarray, output_path: str | Path) -> None:
    expert = np.asarray(expert, dtype=np.float32)
    generated = np.asarray(generated, dtype=np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(expert[:, 0], expert[:, 1], expert[:, 2], label="expert")
    ax.plot(generated[:, 0], generated[:, 1], generated[:, 2], label="TGOD-SD")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
