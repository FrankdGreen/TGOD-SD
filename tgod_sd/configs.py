from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvConfig:
    scene_xml: str = "universal_robots_ur5e/scene.xml"
    ur5e_xml: str = "universal_robots_ur5e/ur5e.xml"

    # 专家轨迹500帧，对应499次状态转移
    horizon: int = 499
    frame_skip: int = 20

    # joint_delta / cartesian_delta
    action_mode: str = "cartesian_delta"

    # 旧关节动作模式使用
    action_scale: float = 0.02

    # 笛卡尔动作模式使用
    position_action_scale: float = 0.015
    rotation_action_scale: float = 0.02

    # 阻尼雅可比参数
    ik_damping: float = 0.03
    ik_orientation_weight: float = 0.0
    max_joint_delta: float = 0.02
    # 单次笛卡尔动作内部最多进行多少次IK迭代
    ik_max_iterations: int = 20
    # IK末端位置误差低于该值时停止迭代，单位m
    ik_position_tolerance: float = 1e-4
    # 每次IK迭代采用多少比例的关节增量
    ik_step_gain: float = 0.8

    # 防止增量动作连续累积后，控制参考远离机器人实际状态。
    max_tcp_reference_error: float = 0.08
    max_joint_target_error: float = 0.18
    tracking_error_scale: float = 0.05

    site_name: str = "attachment_site"
    patch_mesh_assets: str = "never"

    reset_noise: float = 0.0
    initial_state_path: str | None = (
        "data/expert_initial_state.npz"
    )
    position_only: bool = True


@dataclass
class SACConfig:
    hidden_dim: int = 256
    gamma: float = 0.98
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    alpha_init: float = 0.05
    target_entropy: float | None = None


@dataclass
class TGODConfig:
    z_dim: int = 4
    lr_mine: float = 1e-4

    # tcp_tracking / tgod
    reward_mode: str = "tcp_tracking"

    reward_scale: float = 1.0
    reward_clip: float = 2.0
    anchor_reward_weight: float = 0.0
    mine_zs_weight: float = 1.0
    mine_zd_weight: float = 1.0

    # TCP跟踪基准奖励参数
    position_error_scale: float = 0.10
    orientation_error_scale: float = 0.20
    position_reward_weight: float = 1.0
    orientation_reward_weight: float = 0.0
    action_penalty_weight: float = 0.001

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
    eval_interval: int = 10
    save_interval: int = 100
    device: str = "auto"
    fixed_z: bool = False


@dataclass
class MatchConfig:
    n_candidates: int = 20
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iters: int = 100
    deterministic: bool = True
