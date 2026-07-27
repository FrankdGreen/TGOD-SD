from __future__ import annotations

import numpy as np

from tgod_sd.configs import EnvConfig
from tgod_sd.envs.ur5e_env import UR5eTGODEnv


def main() -> None:
    env = UR5eTGODEnv(
        EnvConfig(
            scene_xml="data/scene.xml",
            ur5e_xml="data/ur5e.xml",

            horizon=20,
            frame_skip=5,

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

        reference_before = (
            env.tcp_reference.copy()
        )

        # 只给Z方向一个+1动作
        action = np.array(
            [0, 0, 1, 0, 0, 0],
            dtype=np.float32,
        )

        env.step(action)

        reference_after_action = (
            env.tcp_reference.copy()
        )

        expected_delta = np.array(
            [0, 0, 0.005],
            dtype=np.float64,
        )

        actual_delta = (
            reference_after_action[:3]
            - reference_before[:3]
        )

        print(
            "期望参考增量：",
            expected_delta,
        )

        print(
            "实际参考增量：",
            actual_delta,
        )

        # 后续零动作不应继续改变参考
        zero_action = np.zeros(
            6,
            dtype=np.float32,
        )

        for _ in range(10):
            env.step(zero_action)

        reference_after_zero = (
            env.tcp_reference.copy()
        )

        zero_action_drift = (
            reference_after_zero
            - reference_after_action
        )

        print(
            "10次零动作后的参考变化：",
            zero_action_drift,
        )

        assert np.allclose(
            actual_delta,
            expected_delta,
            atol=1e-10,
        ), (
            "单步TCP参考增量不正确"
        )

        assert np.allclose(
            zero_action_drift,
            0.0,
            atol=1e-10,
        ), (
            "零动作期间TCP参考发生了变化"
        )

        print("参考更新逻辑验证通过")

    finally:
        env.close()


if __name__ == "__main__":
    main()