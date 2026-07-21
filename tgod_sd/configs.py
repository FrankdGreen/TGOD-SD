from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvConfig:
    scene_xml: str = "universal_robots_ur5e/scene.xml"
    ur5e_xml: str = "universal_robots_ur5e/ur5e.xml"

    horizon: int = 500
    frame_skip: int = 5
    action_scale: float = 0.05
    site_name: str = "attachment_site"
    patch_mesh_assets: str = "never"

    # 固定专家起点时必须为0
    reset_noise: float = 0.0

    # 新增：专家初始状态文件
    initial_state_path: str | None = (
        "data/expert_initial_state.npz"
    )


@dataclass
class SACConfig:
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    alpha_init: float = 0.2
    target_entropy: float | None = None


@dataclass
class TGODConfig:
    z_dim: int = 16
    lr_mine: float = 1e-4
    reward_scale: float = 1.0
    reward_clip: float = 20.0
    # 工程稳定项：不是论文 TGOD 核心公式。调试时可以给 0.01~0.05。
    anchor_reward_weight: float = 0.0
    reward_mode: str = "tracking"


@dataclass
class TrainConfig:
    expert_demo: str = "data/expert_demo.npy"
    output_dir: str = "outputs"
    seed: int = 0
    episodes: int = 2000
    updates_per_step: int = 1
    batch_size: int = 256
    replay_size: int = 1_000_000

    # 论文形式建议设为0：
    # 从一开始就由随机SAC策略采样动作
    start_steps: int = 0

    # 新增：经验池达到一定规模后才更新网络
    learning_starts: int = 2500

    log_interval: int = 10
    save_interval: int = 100
    device: str = "auto"


@dataclass
class MatchConfig:
    n_candidates: int = 20
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iters: int = 100
    deterministic: bool = True
