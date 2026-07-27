from __future__ import annotations

import numpy as np

from tgod_sd.configs import EnvConfig
from tgod_sd.envs.ur5e_env import UR5eTGODEnv


def main() -> None:
    env = UR5eTGODEnv(
        EnvConfig(
            scene_xml="data/scene.xml",
            ur5e_xml="data/ur5e.xml",

            horizon=500,
            frame_skip=10,

            action_mode="cartesian_delta",

            position_action_scale=0.005,
            rotation_action_scale=0.02,

            ik_damping=0.03,
            ik_orientation_weight=0.0,
            max_joint_delta=0.02,

            ik_max_iterations=20,
            ik_position_tolerance=1e-4,
            ik_step_gain=0.8,

            reset_noise=0.0,
            initial_state_path=(
                "data/expert_initial_state.npz"
            ),

            patch_mesh_assets="never",
        )
    )

    try:
        env.reset()

        # 保持测试中先排除初始速度影响
        env.data.qvel[:] = 0.0

        env.mujoco.mj_forward(
            env.model,
            env.data,
        )

        # 重新以当前静止状态初始化参考
        env.q_reference = np.asarray(
            env.data.qpos[env.qpos_ids],
            dtype=np.float64,
        ).copy()

        env.tcp_reference = np.asarray(
            env.tcp_feature()[:6],
            dtype=np.float64,
        ).copy()

        env.data.ctrl[:] = (
            env.q_reference
        )

        q_reference_start = (
            env.q_reference.copy()
        )

        tcp_reference_start = (
            env.tcp_reference.copy()
        )

        tcp_start = (
            env.tcp_feature().copy()
        )

        zero_action = np.zeros(
            env.action_dim,
            dtype=np.float32,
        )

        tcp_history = [
            tcp_start.copy()
        ]

        for _ in range(500):
            _, _, done, info = env.step(
                zero_action
            )

            tcp_history.append(
                info["tcp"].copy()
            )

            if done:
                break

        tcp_history = np.asarray(
            tcp_history,
            dtype=np.float64,
        )

        tcp_end = tcp_history[-1]

        actual_drift = (
            tcp_end[:3]
            - tcp_start[:3]
        )

        q_reference_drift = (
            env.q_reference
            - q_reference_start
        )

        tcp_reference_drift = (
            env.tcp_reference
            - tcp_reference_start
        )

        print("=" * 70)
        print("持久参考保持测试")
        print("=" * 70)

        print(
            "初始实际TCP位置：",
            tcp_start[:3],
        )

        print(
            "最终实际TCP位置：",
            tcp_end[:3],
        )

        print(
            "实际XYZ漂移：",
            actual_drift,
        )

        print(
            "实际总漂移：",
            np.linalg.norm(
                actual_drift
            ),
            "m",
        )

        print(
            "q_reference变化：",
            q_reference_drift,
        )

        print(
            "q_reference最大变化：",
            np.max(
                np.abs(
                    q_reference_drift
                )
            ),
        )

        print(
            "tcp_reference变化：",
            tcp_reference_drift,
        )

        print(
            "tcp_reference最大变化：",
            np.max(
                np.abs(
                    tcp_reference_drift
                )
            ),
        )

        np.savez(
            "outputs_hold_reference.npz",

            tcp=tcp_history.astype(
                np.float32
            ),

            q_reference_start=(
                q_reference_start.astype(
                    np.float32
                )
            ),

            q_reference_end=(
                env.q_reference.astype(
                    np.float32
                )
            ),

            tcp_reference_start=(
                tcp_reference_start.astype(
                    np.float32
                )
            ),

            tcp_reference_end=(
                env.tcp_reference.astype(
                    np.float32
                )
            ),
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()