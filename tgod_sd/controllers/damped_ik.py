from __future__ import annotations

import numpy as np


def solve_position_target_iterative(
    *,
    mujoco,
    model,
    ik_data,
    site_id: int,
    qpos_ids: np.ndarray,
    dof_ids: np.ndarray,
    current_qpos: np.ndarray,
    target_position: np.ndarray,
    damping: float,
    max_iterations: int,
    position_tolerance: float,
    max_joint_step: float,
    step_gain: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> dict[str, object]:
    """
    通过多次阻尼雅可比迭代，把末端目标位置转换成关节目标角。

    注意：
    1. 该函数使用独立的ik_data，不修改真实MuJoCo动力学状态；
    2. 当前只求解末端位置，不求解末端姿态；
    3. 返回的是关节目标q_target，不直接推进物理仿真。
    """

    qpos_ids = np.asarray(
        qpos_ids,
        dtype=np.int64,
    )

    dof_ids = np.asarray(
        dof_ids,
        dtype=np.int64,
    )

    q_work = np.asarray(
        current_qpos,
        dtype=np.float64,
    ).copy()

    target_position = np.asarray(
        target_position,
        dtype=np.float64,
    ).reshape(3)

    joint_lower = np.asarray(
        joint_lower,
        dtype=np.float64,
    ).reshape(-1)

    joint_upper = np.asarray(
        joint_upper,
        dtype=np.float64,
    ).reshape(-1)

    damping = max(
        float(damping),
        1e-8,
    )

    max_iterations = max(
        int(max_iterations),
        1,
    )

    position_tolerance = max(
        float(position_tolerance),
        0.0,
    )

    max_joint_step = max(
        float(max_joint_step),
        1e-8,
    )

    step_gain = float(
        np.clip(
            step_gain,
            0.0,
            1.0,
        )
    )

    jac_pos = np.zeros(
        (3, model.nv),
        dtype=np.float64,
    )

    jac_rot = np.zeros(
        (3, model.nv),
        dtype=np.float64,
    )

    final_error = float("inf")
    condition_number = float("inf")
    iterations_used = 0

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        iterations_used = iteration

        # 把当前迭代关节角写入专门的IK数据
        ik_data.qpos[:] = q_work
        ik_data.qvel[:] = 0.0

        mujoco.mj_forward(
            model,
            ik_data,
        )

        current_position = np.asarray(
            ik_data.site_xpos[site_id],
            dtype=np.float64,
        ).copy()

        position_error = (
            target_position
            - current_position
        )

        final_error = float(
            np.linalg.norm(
                position_error
            )
        )

        if final_error <= position_tolerance:
            break

        mujoco.mj_jacSite(
            model,
            ik_data,
            jac_pos,
            jac_rot,
            site_id,
        )

        # 位置模式只使用3×N位置雅可比
        jacobian = jac_pos[:, dof_ids]

        condition_number = float(
            np.linalg.cond(
                jacobian
            )
        )

        regularized = (
            jacobian @ jacobian.T
            + damping**2
            * np.eye(
                3,
                dtype=np.float64,
            )
        )

        try:
            delta_q = (
                jacobian.T
                @ np.linalg.solve(
                    regularized,
                    position_error,
                )
            )
        except np.linalg.LinAlgError:
            delta_q = (
                np.linalg.pinv(
                    jacobian,
                    rcond=1e-6,
                )
                @ position_error
            )

        # 限制每次IK内部迭代的关节变化
        largest_delta = float(
            np.max(
                np.abs(delta_q)
            )
        )

        if largest_delta > max_joint_step:
            delta_q *= (
                max_joint_step
                / largest_delta
            )

        q_work[qpos_ids] += (
            step_gain
            * delta_q
        )

        # 按控制范围限制关节目标
        q_work[qpos_ids] = np.clip(
            q_work[qpos_ids],
            joint_lower,
            joint_upper,
        )

    # 最后重新计算一次真实残差
    ik_data.qpos[:] = q_work
    ik_data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        ik_data,
    )

    final_position = np.asarray(
        ik_data.site_xpos[site_id],
        dtype=np.float64,
    ).copy()

    final_error = float(
        np.linalg.norm(
            target_position
            - final_position
        )
    )

    q_target = q_work[
        qpos_ids
    ].copy()

    return {
        "q_target": q_target,
        "position_error": final_error,
        "iterations": iterations_used,
        "condition_number": condition_number,
        "final_position": final_position,
    }