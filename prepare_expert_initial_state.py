from pathlib import Path

import numpy as np

from tgod_sd.configs import EnvConfig
from tgod_sd.envs.ur5e_env import UR5eTGODEnv


def main() -> None:
    expert_path = Path("data/expert_demo.npy")
    expert_qpos_path = Path("data/expert_qpos.npy")
    output_path = Path("data/expert_initial_state.npz")

    expert = np.asarray(
        np.load(expert_path),
        dtype=np.float64,
    )
    expert_qpos = np.asarray(
        np.load(expert_qpos_path),
        dtype=np.float64,
    )

    if expert.ndim != 2 or expert.shape[1] < 12:
        raise ValueError(
            f"专家轨迹应为 (N, 12)，实际为 {expert.shape}"
        )

    if expert_qpos.ndim != 2:
        raise ValueError(
            f"专家关节轨迹应为二维数组，实际为 {expert_qpos.shape}"
        )

    env = UR5eTGODEnv(
        EnvConfig(
            scene_xml="data/scene.xml",
            ur5e_xml="data/ur5e.xml",
            patch_mesh_assets="never",
            reset_noise=0.0,
        )
    )

    try:
        qpos0 = expert_qpos[0, : env.nq].copy()

        env.data.qpos[:] = qpos0
        env.data.qvel[:] = 0.0
        env.mujoco.mj_forward(env.model, env.data)

        jac_pos = np.zeros(
            (3, env.nv),
            dtype=np.float64,
        )
        jac_rot = np.zeros(
            (3, env.nv),
            dtype=np.float64,
        )

        env.mujoco.mj_jacSite(
            env.model,
            env.data,
            jac_pos,
            jac_rot,
            env.site_id,
        )

        # 专家数据顺序：
        # [vx, vy, vz, wx, wy, wz]
        target_twist = np.concatenate(
            [
                expert[0, 6:9],
                expert[0, 9:12],
            ]
        )

        jacobian = np.vstack(
            [
                jac_pos,
                jac_rot,
            ]
        )

        damping = 1e-4
        regularized = (
            jacobian @ jacobian.T
            + damping**2 * np.eye(6)
        )

        qvel0 = jacobian.T @ np.linalg.solve(
            regularized,
            target_twist,
        )

        predicted_twist = jacobian @ qvel0
        twist_error = np.linalg.norm(
            predicted_twist - target_twist
        )

        np.savez(
            output_path,
            qpos=qpos0.astype(np.float32),
            qvel=qvel0.astype(np.float32),
            expert_tcp0=expert[0].astype(np.float32),
        )

        print(f"已保存：{output_path}")
        print("初始关节角：", qpos0)
        print("初始关节速度：", qvel0)
        print("目标TCP速度：", target_twist)
        print("重构TCP速度：", predicted_twist)
        print(f"速度重构误差：{twist_error:.6e}")

    finally:
        env.close()


if __name__ == "__main__":
    main()