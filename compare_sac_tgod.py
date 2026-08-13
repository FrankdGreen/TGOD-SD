from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from tgod_sd.configs import (
    EnvConfig,
    MatchConfig,
    SACConfig,
    TGODConfig,
    TrainConfig,
)
from tgod_sd.training.trainer import TGODSDTrainer
from tgod_sd.utils import ensure_dir, json_dumps


def evaluate_checkpoint(path: str, n_candidates: int) -> dict:
    checkpoint_path = Path(path)
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    trainer = TGODSDTrainer(
        EnvConfig(**state["env_config"]),
        SACConfig(**state["sac_config"]),
        TGODConfig(**state["tgod_config"]),
        TrainConfig(**state["train_config"]),
        MatchConfig(**state["match_config"]),
    )
    try:
        trainer.agent.load_state_dict(state["agent"])
        suite = trainer.evaluate_candidate_suite(n_candidates)
        rmses = suite["rmses"]
        # 10 cm以内的候选才视为围绕专家的可用候选。
        usable = rmses <= 0.10
        metrics = {
            **suite["metrics"],
            "mode": trainer.tgod_cfg.reward_mode,
            "candidate_count": int(len(rmses)),
            "usable_candidate_count_rmse_le_10cm": int(usable.sum()),
            "usable_candidate_rate_rmse_le_10cm": float(usable.mean()),
        }
        return {
            "metrics": metrics,
            "trajectories": suite["trajectories"],
            "rmses": rmses,
            "sinkhorn_values": suite["sinkhorn_values"],
            "latents": suite["latents"],
        }
    finally:
        trainer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用固定候选协议公平比较SAC与TGOD/MINE。"
    )
    parser.add_argument("--sac_checkpoint", required=True)
    parser.add_argument("--tgod_checkpoint", required=True)
    parser.add_argument("--n_candidates", type=int, default=8)
    parser.add_argument("--output_dir", default="outputs_compare_report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    sac = evaluate_checkpoint(args.sac_checkpoint, args.n_candidates)
    tgod = evaluate_checkpoint(args.tgod_checkpoint, args.n_candidates)

    report = {
        "protocol": {
            "n_candidates": args.n_candidates,
            "deterministic_policy": True,
            "tgod_latents": "fixed Gaussian vectors generated from seed+2025",
            "usable_candidate_threshold_rmse_m": 0.10,
        },
        "sac": sac["metrics"],
        "tgod": tgod["metrics"],
    }
    (output_dir / "comparison_metrics.json").write_text(
        json_dumps(report),
        encoding="utf-8",
    )
    np.savez(
        output_dir / "comparison_candidates.npz",
        sac_trajectories=sac["trajectories"],
        sac_rmse=sac["rmses"],
        sac_sinkhorn=sac["sinkhorn_values"],
        sac_latents=sac["latents"],
        tgod_trajectories=tgod["trajectories"],
        tgod_rmse=tgod["rmses"],
        tgod_sinkhorn=tgod["sinkhorn_values"],
        tgod_latents=tgod["latents"],
    )

    expert = np.load("data/expert_demo.npy")
    figure = plt.figure(figsize=(13, 6))
    for panel, title, result in [
        (1, "SAC position tracking", sac),
        (2, "TGOD/MINE candidates", tgod),
    ]:
        axis = figure.add_subplot(1, 2, panel, projection="3d")
        axis.plot(
            expert[:, 0], expert[:, 1], expert[:, 2],
            color="black", linewidth=2.5, label="expert",
        )
        for index, trajectory in enumerate(result["trajectories"]):
            axis.plot(
                trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                linewidth=1.0, alpha=0.75,
                label="candidate" if index == 0 else None,
            )
        axis.set_title(title)
        axis.set_xlabel("x / m")
        axis.set_ylabel("y / m")
        axis.set_zlabel("z / m")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "candidate_comparison.png", dpi=180)
    plt.close(figure)
    print(json_dumps(report))


if __name__ == "__main__":
    main()
