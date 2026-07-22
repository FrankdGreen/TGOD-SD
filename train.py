from __future__ import annotations

import argparse

from tgod_sd.configs import EnvConfig, MatchConfig, SACConfig, TGODConfig, TrainConfig
from tgod_sd.training.trainer import TGODSDTrainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TGOD-SD modular reproduction for UR5e + MuJoCo")

    # data / output
    p.add_argument("--expert_demo", type=str, default="data/expert_demo.npy")
    p.add_argument("--scene_xml", type=str, default="data/scene.xml")
    p.add_argument("--ur5e_xml", type=str, default="data/ur5e.xml")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")

    # env
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--frame_skip", type=int, default=5)
    p.add_argument("--action_scale", type=float, default=0.05)
    p.add_argument("--site_name", type=str, default="attachment_site")
    p.add_argument("--patch_mesh_assets", type=str, default="auto", choices=["auto", "always", "never"])
    p.add_argument("--reset_noise", type=float, default=0.01)
    p.add_argument("--initial_state_path", type=str, default="data/expert_initial_state.npz", help="每个episode共同使用的专家初始状态。")

    # train
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--updates_per_step", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--replay_size", type=int, default=1_000_000)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument(
        "--start_steps",
        type=int,
        default=0,
        help=(
            "完全均匀随机动作步数。"
            "论文一致模式建议为0。"
        ),
    )
    p.add_argument(
        "--learning_starts",
        type=int,
        default=2500,
        help="经验池达到该数量后才开始更新网络。",
    )
    p.add_argument(
        "--fixed_z",
        action="store_true",
        help="所有episode固定使用零向量z，用于纯轨迹跟踪基准。",
    )

    # SAC
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr_actor", type=float, default=3e-4)
    p.add_argument("--lr_critic", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=3e-4)
    p.add_argument("--alpha_init", type=float, default=0.2)

    # TGOD / MINE
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--lr_mine", type=float, default=1e-4)
    p.add_argument("--reward_scale", type=float, default=1.0)
    p.add_argument("--reward_clip", type=float, default=20.0)
    p.add_argument("--anchor_reward_weight", type=float, default=0.05)

    # matching
    p.add_argument("--n_candidates", type=int, default=20)
    p.add_argument("--sinkhorn_epsilon", type=float, default=0.1)
    p.add_argument("--sinkhorn_iters", type=int, default=100)
    p.add_argument("--stochastic_match", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    env_cfg = EnvConfig(
        scene_xml=args.scene_xml,
        ur5e_xml=args.ur5e_xml,
        horizon=args.horizon,
        frame_skip=args.frame_skip,
        action_scale=args.action_scale,
        site_name=args.site_name,
        patch_mesh_assets=args.patch_mesh_assets,
        reset_noise=args.reset_noise,
        initial_state_path=args.initial_state_path,
    )

    sac_cfg = SACConfig(
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        tau=args.tau,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        alpha_init=args.alpha_init,
    )
    tgod_cfg = TGODConfig(
        z_dim=args.z_dim,
        lr_mine=args.lr_mine,
        reward_scale=args.reward_scale,
        reward_clip=args.reward_clip,
        anchor_reward_weight=args.anchor_reward_weight,
    )
    train_cfg = TrainConfig(
        expert_demo=args.expert_demo,
        output_dir=args.output_dir,
        seed=args.seed,
        episodes=args.episodes,
        updates_per_step=args.updates_per_step,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        start_steps=args.start_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
        learning_starts=args.learning_starts,
        fixed_z=args.fixed_z,
    )
    match_cfg = MatchConfig(
        n_candidates=args.n_candidates,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        deterministic=not args.stochastic_match,
    )

    trainer = TGODSDTrainer(env_cfg, sac_cfg, tgod_cfg, train_cfg, match_cfg)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
