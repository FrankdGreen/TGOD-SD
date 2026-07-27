from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tgod_sd.configs import EnvConfig
from tgod_sd.envs.ur5e_env import UR5eTGODEnv
from tgod_sd.utils import (
    ensure_dir,
    load_expert_demo,
    rotation_error_rotvec,
)


def trajectory_length(xyz: np.ndarray) -> float:
    """计算三维轨迹的总长度。"""
    xyz = np.asarray(xyz, dtype=np.float64)

    if len(xyz) < 2:
        return 0.0

    delta = np.diff(xyz, axis=0)

    return float(
        np.sum(
            np.linalg.norm(
                delta,
                axis=1,
            )
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "验证笛卡尔末端动作经过阻尼雅可比控制器后，"
            "能否在MuJoCo中跟踪专家TCP轨迹。"
        )
    )

    parser.add_argument(
        "--expert_demo",
        type=str,
        default="data/expert_demo.npy",
    )
    parser.add_argument(
        "--ik_max_iterations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--ik_position_tolerance",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--ik_step_gain",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--scene_xml",
        type=str,
        default="data/scene.xml",
    )

    parser.add_argument(
        "--ur5e_xml",
        type=str,
        default="data/ur5e.xml",
    )

    parser.add_argument(
        "--initial_state_path",
        type=str,
        default="data/expert_initial_state.npz",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_cartesian_controller_test",
    )

    parser.add_argument(
        "--frame_skip",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--position_action_scale",
        type=float,
        default=0.005,
        help="Actor归一化动作1.0对应的最大单步位置增量，单位m。",
    )

    parser.add_argument(
        "--rotation_action_scale",
        type=float,
        default=0.02,
        help="Actor归一化动作1.0对应的最大单步旋转增量，单位rad。",
    )

    parser.add_argument(
        "--ik_damping",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--ik_orientation_weight",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--max_joint_delta",
        type=float,
        default=0.02,
        help="雅可比控制器输出的单步最大关节增量，单位rad。",
    )

    parser.add_argument(
        "--kp_position",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--kp_orientation",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--patch_mesh_assets",
        type=str,
        default="never",
        choices=[
            "auto",
            "always",
            "never",
        ],
    )
    parser.add_argument(
        "--position_only",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_dir = ensure_dir(
        args.output_dir
    )

    expert = load_expert_demo(
        args.expert_demo
    ).astype(np.float64)

    if expert.shape[0] < 2:
        raise ValueError(
            "专家轨迹至少需要2帧。"
        )

    print(
        "[EXPERT]",
        "shape =",
        expert.shape,
    )

    # 专家有500帧时，只需要499次状态转移：
    # expert[0] -> expert[1] -> ... -> expert[499]
    horizon = len(expert) - 1

    env_cfg = EnvConfig(
        scene_xml=args.scene_xml,
        ur5e_xml=args.ur5e_xml,
        horizon=horizon,
        frame_skip=args.frame_skip,

        # 本测试必须使用笛卡尔动作模式
        action_mode="cartesian_delta",

        position_action_scale=(
            args.position_action_scale
        ),
        rotation_action_scale=(
            args.rotation_action_scale
        ),

        ik_damping=args.ik_damping,
        ik_orientation_weight=0.0,
        max_joint_delta=(
            args.max_joint_delta
        ),

        site_name="attachment_site",
        patch_mesh_assets=(
            args.patch_mesh_assets
        ),

        reset_noise=0.0,
        initial_state_path=(
            args.initial_state_path
        ),
        ik_max_iterations=(
            args.ik_max_iterations
        ),

        ik_position_tolerance=(
            args.ik_position_tolerance
        ),

        ik_step_gain=(
            args.ik_step_gain
        ),
    )

    env = UR5eTGODEnv(env_cfg)

    actual_tcp_list: list[np.ndarray] = []
    actual_qpos_list: list[np.ndarray] = []

    action_list: list[np.ndarray] = []
    cartesian_delta_list: list[np.ndarray] = []
    joint_delta_list: list[np.ndarray] = []
    q_target_list: list[np.ndarray] = []
    condition_list: list[float] = []
    ik_error_list: list[float] = []
    ik_iteration_list: list[int] = []
    servo_error_list: list[np.ndarray] = []
    position_error_list: list[float] = []
    orientation_error_list: list[float] = []

    try:
        env.reset()

        # 保存真正的初始状态，对应 expert[0]
        initial_tcp = env.tcp_feature().copy()
        initial_qpos = (
            env.data.qpos.copy()
        )

        actual_tcp_list.append(
            initial_tcp
        )
        actual_qpos_list.append(
            initial_qpos
        )

        initial_position_error = float(
            np.linalg.norm(
                initial_tcp[:3]
                - expert[0, :3]
            )
        )

        initial_orientation_error = (
            rotation_error_rotvec(
                current_rotvec=(
                    initial_tcp[3:6]
                ),
                target_rotvec=(
                    expert[0, 3:6]
                ),
            )
        )

        print(
            "[INITIAL]",
            f"position_error="
            f"{initial_position_error:.8f} m",
        )

        print(
            "[INITIAL]",
            f"orientation_error="
            f"{np.linalg.norm(initial_orientation_error):.8f} rad",
        )

        for target_index in range(
            1,
            len(expert),
        ):
            current_tcp = (
                env.tcp_feature()
                .astype(np.float64)
            )

            target_tcp = expert[
                target_index
            ]

            if env.tcp_reference is None:
                raise RuntimeError(
                    "环境中的tcp_reference没有初始化"
                )

            # 注意这里比较的是专家目标和“控制参考”，
            # 不是专家目标和实际TCP。
            reference_position_error = (
                    target_tcp[:3]
                    - env.tcp_reference[:3]
            )

            position_action = (
                    args.kp_position
                    * reference_position_error
                    / args.position_action_scale
            )

            # 世界坐标系中的姿态误差
            orientation_error = (
                rotation_error_rotvec(
                    current_rotvec=(
                        current_tcp[3:6]
                    ),
                    target_rotvec=(
                        target_tcp[3:6]
                    ),
                ).astype(np.float64)
            )

            if args.position_only:
                orientation_action = np.zeros(
                    3,
                    dtype=np.float64,
                )
            else:
                orientation_action = (
                        args.kp_orientation
                        * orientation_error
                        / args.rotation_action_scale
                )

            action = np.concatenate(
                [
                    position_action,
                    orientation_action,
                ]
            )

            action = np.clip(
                action,
                -1.0,
                1.0,
            ).astype(np.float32)

            (
                _,
                _,
                done,
                info,
            ) = env.step(action)

            required_keys = {
                "tcp",
                "qpos",
                "cartesian_delta",
                "joint_delta",
                "q_target",
                "jacobian_condition",
            }

            missing_keys = (
                required_keys
                - set(info.keys())
            )

            if missing_keys:
                raise RuntimeError(
                    "当前ur5e_env.py还没有完整实现"
                    "笛卡尔动作模式，info中缺少："
                    f"{sorted(missing_keys)}"
                )

            actual_tcp = np.asarray(
                info["tcp"],
                dtype=np.float64,
            )

            actual_qpos = np.asarray(
                info["qpos"],
                dtype=np.float64,
            )

            actual_tcp_list.append(
                actual_tcp
            )

            actual_qpos_list.append(
                actual_qpos
            )

            action_list.append(
                action.copy()
            )

            cartesian_delta_list.append(
                np.asarray(
                    info["cartesian_delta"],
                    dtype=np.float64,
                )
            )

            joint_delta_list.append(
                np.asarray(
                    info["joint_delta"],
                    dtype=np.float64,
                )
            )

            q_target_list.append(
                np.asarray(
                    info["q_target"],
                    dtype=np.float64,
                )
            )

            condition_list.append(
                float(
                    info[
                        "jacobian_condition"
                    ]
                )
            )
            ik_error_list.append(
                float(
                    info["ik_position_error"]
                )
            )

            ik_iteration_list.append(
                int(
                    info["ik_iterations"]
                )
            )

            servo_error_list.append(
                np.asarray(
                    info["servo_error"],
                    dtype=np.float64,
                )
            )

            position_error_after = float(
                np.linalg.norm(
                    actual_tcp[:3]
                    - target_tcp[:3]
                )
            )

            orientation_error_after = (
                rotation_error_rotvec(
                    current_rotvec=(
                        actual_tcp[3:6]
                    ),
                    target_rotvec=(
                        target_tcp[3:6]
                    ),
                )
            )

            position_error_list.append(
                position_error_after
            )

            orientation_error_list.append(
                float(
                    np.linalg.norm(
                        orientation_error_after
                    )
                )
            )

            # 前5步打印详细信息，确认控制方向正确
            if target_index <= 5:
                print(
                    "\n[STEP]",
                    target_index,
                )

                print(
                    "target xyz =",
                    target_tcp[:3],
                )

                print(
                    "actual xyz =",
                    actual_tcp[:3],
                )

                print(
                    "position error =",
                    position_error_after,
                )

                print(
                    "orientation error =",
                    orientation_error_list[-1],
                )

                print(
                    "normalized action =",
                    action,
                )

                print(
                    "joint delta =",
                    joint_delta_list[-1],
                )

                print(
                    "Jacobian condition =",
                    condition_list[-1],
                )

            if done and target_index < (
                len(expert) - 1
            ):
                print(
                    "[WARN]",
                    "环境提前结束：",
                    target_index,
                )
                break

    finally:
        env.close()

    actual_tcp = np.asarray(
        actual_tcp_list,
        dtype=np.float64,
    )

    actual_qpos = np.asarray(
        actual_qpos_list,
        dtype=np.float64,
    )

    actions = np.asarray(
        action_list,
        dtype=np.float64,
    )

    cartesian_deltas = np.asarray(
        cartesian_delta_list,
        dtype=np.float64,
    )

    joint_deltas = np.asarray(
        joint_delta_list,
        dtype=np.float64,
    )

    q_targets = np.asarray(
        q_target_list,
        dtype=np.float64,
    )

    condition_numbers = np.asarray(
        condition_list,
        dtype=np.float64,
    )

    compare_length = min(
        len(actual_tcp),
        len(expert),
    )

    actual_tcp = actual_tcp[
        :compare_length
    ]

    actual_qpos = actual_qpos[
        :compare_length
    ]

    expert_compare = expert[
        :compare_length
    ]

    position_difference = (
        actual_tcp[:, :3]
        - expert_compare[:, :3]
    )

    position_distance = np.linalg.norm(
        position_difference,
        axis=1,
    )

    position_rmse = float(
        np.sqrt(
            np.mean(
                position_distance**2
            )
        )
    )

    orientation_distances = []

    for index in range(compare_length):
        error = rotation_error_rotvec(
            current_rotvec=(
                actual_tcp[index, 3:6]
            ),
            target_rotvec=(
                expert_compare[
                    index,
                    3:6,
                ]
            ),
        )

        orientation_distances.append(
            np.linalg.norm(error)
        )

    orientation_distances = np.asarray(
        orientation_distances,
        dtype=np.float64,
    )

    orientation_rmse = float(
        np.sqrt(
            np.mean(
                orientation_distances**2
            )
        )
    )

    generated_length = trajectory_length(
        actual_tcp[:, :3]
    )

    expert_length = trajectory_length(
        expert_compare[:, :3]
    )

    path_length_ratio = (
        generated_length
        / max(expert_length, 1e-12)
    )

    action_saturation = (
        float(
            np.mean(
                np.abs(actions) > 0.95
            )
        )
        if actions.size
        else 0.0
    )

    joint_delta_saturation = (
        float(
            np.mean(
                np.abs(joint_deltas)
                >= (
                    args.max_joint_delta
                    * 0.99
                )
            )
        )
        if joint_deltas.size
        else 0.0
    )

    finite_conditions = (
        condition_numbers[
            np.isfinite(
                condition_numbers
            )
        ]
    )

    mean_condition = (
        float(
            np.mean(
                finite_conditions
            )
        )
        if finite_conditions.size
        else float("inf")
    )

    max_condition = (
        float(
            np.max(
                finite_conditions
            )
        )
        if finite_conditions.size
        else float("inf")
    )

    ik_errors = np.asarray(
        ik_error_list,
        dtype=np.float64,
    )

    ik_iterations = np.asarray(
        ik_iteration_list,
        dtype=np.int32,
    )

    servo_errors = np.asarray(
        servo_error_list,
        dtype=np.float64,
    )

    mean_ik_error = float(
        np.mean(ik_errors)
    )

    max_ik_error = float(
        np.max(ik_errors)
    )

    mean_ik_iterations = float(
        np.mean(ik_iterations)
    )

    servo_rmse = float(
        np.sqrt(
            np.mean(
                servo_errors ** 2
            )
        )
    )

    servo_max = float(
        np.max(
            np.abs(
                servo_errors
            )
        )
    )


    print("\n" + "=" * 70)
    print("笛卡尔控制器测试结果")
    print("=" * 70)

    print(
        f"实际轨迹帧数："
        f"{compare_length}"
    )

    print(
        f"TCP位置RMSE："
        f"{position_rmse:.6f} m"
    )

    print(
        f"TCP姿态RMSE："
        f"{orientation_rmse:.6f} rad"
    )

    print(
        f"专家轨迹长度："
        f"{expert_length:.6f} m"
    )

    print(
        f"生成轨迹长度："
        f"{generated_length:.6f} m"
    )

    print(
        f"轨迹长度比例："
        f"{path_length_ratio:.4f}"
    )

    print(
        f"归一化动作饱和率："
        f"{action_saturation:.2%}"
    )

    print(
        f"关节增量饱和率："
        f"{joint_delta_saturation:.2%}"
    )

    print(
        f"Jacobian平均条件数："
        f"{mean_condition:.3f}"
    )

    print(
        f"Jacobian最大条件数："
        f"{max_condition:.3f}"
    )
    print(
        f"IK平均位置残差："
        f"{mean_ik_error:.6e} m"
    )

    print(
        f"IK最大位置残差："
        f"{max_ik_error:.6e} m"
    )

    print(
        f"IK平均迭代次数："
        f"{mean_ik_iterations:.2f}"
    )

    print(
        f"关节伺服RMSE："
        f"{servo_rmse:.6f} rad"
    )

    print(
        f"关节伺服最大误差："
        f"{servo_max:.6f} rad"
    )

    output_npz = (
        Path(output_dir)
        / "cartesian_controller_test.npz"
    )

    np.savez(
        output_npz,

        tcp=actual_tcp.astype(
            np.float32
        ),

        qpos=actual_qpos.astype(
            np.float32
        ),

        action=actions.astype(
            np.float32
        ),

        cartesian_delta=(
            cartesian_deltas.astype(
                np.float32
            )
        ),

        joint_delta=(
            joint_deltas.astype(
                np.float32
            )
        ),

        q_target=q_targets.astype(
            np.float32
        ),

        jacobian_condition=(
            condition_numbers.astype(
                np.float32
            )
        ),

        expert_demo=expert_compare.astype(
            np.float32
        ),
        ik_position_error=(
            ik_errors.astype(
                np.float32
            )
        ),

        ik_iterations=(
            ik_iterations.astype(
                np.int32
            )
        ),

        servo_error=(
            servo_errors.astype(
                np.float32
            )
        ),
    )

    print(
        f"测试数据已保存："
        f"{output_npz}"
    )

    # 绘制三维TCP轨迹
    figure = plt.figure(
        figsize=(9, 7)
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    axis.plot(
        expert_compare[:, 0],
        expert_compare[:, 1],
        expert_compare[:, 2],
        label="Expert TCP",
    )

    axis.plot(
        actual_tcp[:, 0],
        actual_tcp[:, 1],
        actual_tcp[:, 2],
        label="Cartesian controller",
    )

    axis.scatter(
        expert_compare[0, 0],
        expert_compare[0, 1],
        expert_compare[0, 2],
        marker="o",
        label="Start",
    )

    axis.set_xlabel("X / m")
    axis.set_ylabel("Y / m")
    axis.set_zlabel("Z / m")
    axis.set_title(
        "Expert TCP vs Cartesian Controller"
    )
    axis.legend()

    figure.tight_layout()

    plot_path = (
        Path(output_dir)
        / "expert_vs_cartesian_controller.png"
    )

    figure.savefig(
        plot_path,
        dpi=180,
    )

    plt.close(figure)

    print(
        f"轨迹图片已保存："
        f"{plot_path}"
    )


if __name__ == "__main__":
    main()