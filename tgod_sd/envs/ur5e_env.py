from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from tgod_sd.configs import EnvConfig
from tgod_sd.utils import rotmat_to_rotvec
from tgod_sd.xml_utils import prepare_runtime_xml
from tgod_sd.controllers.damped_ik import (
    solve_position_target_iterative,
)


class UR5eTGODEnv:
    """
    UR5e 的 TGOD-SD 环境封装。

    动作含义：SAC 输出 action in [-1,1]^6，环境将其解释为关节目标角增量：
        q_target = q_current + action_scale * action
    然后把 q_target 写入 MuJoCo ctrl，由 MuJoCo 执行器推进动力学。
    """

    def __init__(self, cfg: EnvConfig):
        try:
            import mujoco  # type: ignore
        except Exception as exc:
            raise RuntimeError("缺少 mujoco，请先执行：pip install mujoco") from exc

        self.cfg = cfg
        self.mujoco = mujoco
        self.horizon = int(cfg.horizon)
        self.frame_skip = int(cfg.frame_skip)
        self.action_mode = cfg.action_mode

        self.action_scale = float(
            cfg.action_scale
        )

        self.position_action_scale = float(
            cfg.position_action_scale
        )

        self.rotation_action_scale = float(
            cfg.rotation_action_scale
        )

        self.ik_damping = float(
            cfg.ik_damping
        )

        self.ik_orientation_weight = float(
            cfg.ik_orientation_weight
        )

        self.max_joint_delta = float(
            cfg.max_joint_delta
        )
        self.ik_max_iterations = int(
            cfg.ik_max_iterations
        )

        self.ik_position_tolerance = float(
            cfg.ik_position_tolerance
        )

        self.ik_step_gain = float(
            cfg.ik_step_gain
        )
        self.max_tcp_reference_error = float(
            cfg.max_tcp_reference_error
        )
        self.max_joint_target_error = float(
            cfg.max_joint_target_error
        )
        self.tracking_error_scale = max(
            float(cfg.tracking_error_scale),
            1e-6,
        )
        self.reset_noise = float(cfg.reset_noise)
        self.t = 0
        self._runtime_tmp: Optional[tempfile.TemporaryDirectory] = None
        # 持久化控制参考。
        # 不能在每一步重新用实际qpos或实际TCP覆盖。
        self.q_reference: np.ndarray | None = None
        self.tcp_reference: np.ndarray | None = None
        self.tracking_target_xyz: np.ndarray | None = None

        self.model, self.data = self._load_model(
            scene_xml=cfg.scene_xml,
            ur5e_xml=cfg.ur5e_xml,
            patch_mesh_assets=cfg.patch_mesh_assets,
        )
        # 专门用于逆运动学迭代。
        # 不使用真实self.data，避免IK计算污染动力学状态。
        self.ik_data = self.mujoco.MjData(
            self.model
        )

        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.actuated_joint_ids = np.asarray(
            self.model.actuator_trnid[:, 0],
            dtype=np.int64,
        )

        self.qpos_ids = np.asarray(
            self.model.jnt_qposadr[
                self.actuated_joint_ids
            ],
            dtype=np.int64,
        )

        self.dof_ids = np.asarray(
            self.model.jnt_dofadr[
                self.actuated_joint_ids
            ],
            dtype=np.int64,
        )

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, cfg.site_name)
        if self.site_id < 0:
            if self.model.nsite <= 0:
                raise ValueError("XML 中找不到 site，请检查 TCP site。")
            self.site_id = 0
            print(f"[WARN] 未找到 site={cfg.site_name}，改用 site id=0。")

        self.home_qpos = self._infer_home_qpos()
        self.ctrl_low, self.ctrl_high = self._infer_ctrl_range()

        (
            self.initial_qpos,
            self.initial_qvel,
        ) = self._load_initial_state(
            cfg.initial_state_path
        )

        if cfg.initial_state_path and self.reset_noise != 0:
            raise ValueError(
                "使用专家固定初始状态时，"
                "reset_noise 必须设置为0。"
            )

        obs = self.reset()
        self.obs_dim = int(obs.shape[0])
        if self.action_mode == "cartesian_delta":
            self.action_dim = 3
        else:
            self.action_dim = self.nu
        self.tcp_dim = 12
        self.tcp_slice = slice(self.nq + self.nv, self.nq + self.nv + 12)

    def _load_model(self, scene_xml: str, ur5e_xml: str, patch_mesh_assets: str):
        mujoco = self.mujoco
        patch = False if patch_mesh_assets == "never" else True if patch_mesh_assets == "always" else False
        try:
            scene_path, tmp = prepare_runtime_xml(scene_xml, ur5e_xml, patch_meshes=patch)
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            self._runtime_tmp = tmp
            return model, data
        except Exception as first_exc:
            if patch_mesh_assets == "never":
                raise
            scene_path, tmp = prepare_runtime_xml(scene_xml, ur5e_xml, patch_meshes=True)
            model = mujoco.MjModel.from_xml_path(str(scene_path))
            data = mujoco.MjData(model)
            self._runtime_tmp = tmp
            print("[XML] 原始 XML 加载失败，已自动删除视觉 mesh 后重试。")
            print(f"[XML] 原始错误：{first_exc}")
            return model, data

    def _infer_home_qpos(self) -> np.ndarray:
        home = np.zeros(self.nq, dtype=np.float64)
        try:
            key_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id >= 0:
                home[:] = np.asarray(self.model.key_qpos[key_id]).reshape(-1)[: self.nq]
        except Exception:
            pass
        return home

    def _load_initial_state(
            self,
            path: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        读取每个episode共同使用的专家初始状态。
        未提供路径时退回XML中的home状态。
        """
        if path is None:
            return (
                self.home_qpos.copy(),
                np.zeros(self.nv, dtype=np.float64),
            )

        state_path = Path(path)

        if not state_path.exists():
            raise FileNotFoundError(
                f"找不到专家初始状态文件：{state_path}"
            )

        with np.load(
                state_path,
                allow_pickle=False,
        ) as state:
            if "qpos" not in state:
                raise KeyError(
                    f"{state_path} 中没有 qpos 字段"
                )

            qpos = np.asarray(
                state["qpos"],
                dtype=np.float64,
            ).reshape(-1)

            if "qvel" in state:
                qvel = np.asarray(
                    state["qvel"],
                    dtype=np.float64,
                ).reshape(-1)
            else:
                qvel = np.zeros(
                    self.nv,
                    dtype=np.float64,
                )

        if qpos.shape[0] < self.nq:
            raise ValueError(
                f"qpos维度不足，需要{self.nq}维，"
                f"实际为{qpos.shape}"
            )

        if qvel.shape[0] < self.nv:
            raise ValueError(
                f"qvel维度不足，需要{self.nv}维，"
                f"实际为{qvel.shape}"
            )

        qpos = qpos[: self.nq].copy()
        qvel = qvel[: self.nv].copy()

        print("[RESET] 使用专家固定初始状态")
        print("[RESET] qpos0 =", qpos)
        print("[RESET] qvel0 =", qvel)

        return qpos, qvel

    def _infer_ctrl_range(self) -> tuple[np.ndarray, np.ndarray]:
        ctrlrange = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        if ctrlrange.shape[0] == self.nu and np.all(ctrlrange[:, 0] < ctrlrange[:, 1]):
            return ctrlrange[:, 0], ctrlrange[:, 1]
        return -np.pi * np.ones(self.nu), np.pi * np.ones(self.nu)

    def close(self) -> None:
        if self._runtime_tmp is not None:
            self._runtime_tmp.cleanup()
            self._runtime_tmp = None

    def reset(self) -> np.ndarray:
        self.t = 0

        # 清除上一个episode留下的动力学状态
        self.mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # 恢复专家初始状态
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel

        # 先进行正运动学更新
        self.mujoco.mj_forward(
            self.model,
            self.data,
        )

        # ---------------------------------------
        # 1. 初始化持久关节目标
        # ---------------------------------------
        self.q_reference = np.asarray(
            self.data.qpos[self.qpos_ids],
            dtype=np.float64,
        ).copy()

        self.q_reference = np.clip(
            self.q_reference,
            self.ctrl_low,
            self.ctrl_high,
        )

        self.data.ctrl[:] = self.q_reference

        # ---------------------------------------
        # 2. 初始化持久TCP目标
        # 当前先保存位置+姿态6维
        # ---------------------------------------
        current_tcp = self.tcp_feature()

        self.tcp_reference = np.asarray(
            current_tcp[:6],
            dtype=np.float64,
        ).copy()
        self.tracking_target_xyz = np.asarray(
            current_tcp[:3],
            dtype=np.float64,
        ).copy()

        # 再更新一次，确保ctrl和状态一致
        self.mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self._get_obs()

    def step(
            self,
            action: np.ndarray,
    ):
        action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(self.action_dim)

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        q_current = self.data.qpos[
            self.qpos_ids
        ].copy()

        if self.action_mode == "joint_delta":
            if self.q_reference is None:
                raise RuntimeError(
                    "q_reference尚未初始化，请先调用env.reset()"
                )

            # 动作增量累加到上一时刻的控制目标，
            # 而不是当前实际关节角。
            self.q_reference = (
                    self.q_reference
                    + self.action_scale * action
            )

            self.q_reference = np.clip(
                self.q_reference,
                self.ctrl_low,
                self.ctrl_high,
            )

            q_target = self.q_reference.copy()

            q_actual = np.asarray(
                self.data.qpos[self.qpos_ids],
                dtype=np.float64,
            )

            delta_q = (
                    q_target
                    - q_actual
            )

            requested_cartesian_delta = np.zeros(
                6,
                dtype=np.float64,
            )

            ik_position_error = 0.0
            ik_iterations = 0
            jacobian_condition = 0.0


        elif self.action_mode == "cartesian_delta":

            if self.tcp_reference is None:
                raise RuntimeError(

                    "tcp_reference尚未初始化，请先调用env.reset()"

                )

            if self.q_reference is None:
                raise RuntimeError(

                    "q_reference尚未初始化，请先调用env.reset()"

                )

            # ---------------------------------------

            # 1. 将归一化动作转换成末端位置增量

            # 当前位置-only测试只使用前三维

            # ---------------------------------------

            requested_cartesian_delta = np.concatenate(

                [

                    self.position_action_scale

                    * action[:3],

                    np.zeros(

                        3,

                        dtype=np.float64,

                    ),

                ]

            )

            # ---------------------------------------

            # 2. 更新持久TCP参考

            # 注意：不是 actual_tcp + delta

            # ---------------------------------------

            self.tcp_reference[:3] = (

                    self.tcp_reference[:3]

                    + requested_cartesian_delta[:3]

            )

            # 增量参考只能在实际TCP附近有限范围内移动，避免随机探索
            # 将持久参考持续推向不可达位置。
            actual_position = self.tcp_feature()[:3].astype(
                np.float64
            )
            reference_offset = (
                self.tcp_reference[:3] - actual_position
            )
            reference_distance = float(
                np.linalg.norm(reference_offset)
            )
            if (
                self.max_tcp_reference_error > 0
                and reference_distance > self.max_tcp_reference_error
            ):
                self.tcp_reference[:3] = (
                    actual_position
                    + reference_offset
                    * (
                        self.max_tcp_reference_error
                        / reference_distance
                    )
                )

            target_position = (

                self.tcp_reference[:3].copy()

            )

            # ---------------------------------------

            # 3. 以上一次关节参考作为IK初值

            # 避免IK解在不同分支之间跳动

            # ---------------------------------------

            ik_seed = self.data.qpos.copy()

            ik_seed[self.qpos_ids] = (

                self.q_reference

            )

            ik_result = solve_position_target_iterative(

                mujoco=self.mujoco,

                model=self.model,

                ik_data=self.ik_data,

                site_id=self.site_id,

                qpos_ids=self.qpos_ids,

                dof_ids=self.dof_ids,

                current_qpos=ik_seed,

                target_position=target_position,

                damping=self.ik_damping,

                max_iterations=self.ik_max_iterations,

                position_tolerance=(

                    self.ik_position_tolerance

                ),

                max_joint_step=(

                    self.max_joint_delta

                ),

                step_gain=self.ik_step_gain,

                joint_lower=self.ctrl_low,

                joint_upper=self.ctrl_high,

            )

            q_target = np.asarray(

                ik_result["q_target"],

                dtype=np.float64,

            )

            q_target = np.clip(

                q_target,

                self.ctrl_low,

                self.ctrl_high,

            )

            # IK内部步长限制不等于最终伺服目标限制。这里再限制目标
            # 相对实际关节角的误差，避免一次下发过远的关节目标。
            if self.max_joint_target_error > 0:
                q_target = np.clip(
                    q_target,
                    q_current - self.max_joint_target_error,
                    q_current + self.max_joint_target_error,
                )

            # ---------------------------------------

            # 4. 保存新的持久关节参考

            # ---------------------------------------

            self.q_reference = q_target.copy()

            q_actual = np.asarray(

                self.data.qpos[self.qpos_ids],

                dtype=np.float64,

            )

            delta_q = (

                    q_target

                    - q_actual

            )

            ik_position_error = float(

                ik_result["position_error"]

            )

            ik_iterations = int(

                ik_result["iterations"]

            )

            jacobian_condition = float(

                ik_result["condition_number"]

            )
        else:
            raise ValueError(
                f"未知action_mode："
                f"{self.action_mode}"
            )

        q_target = (
                q_current
                + delta_q
        )

        self.data.ctrl[:] = np.clip(
            q_target,
            self.ctrl_low,
            self.ctrl_high,
        )

        for _ in range(self.frame_skip):
            self.mujoco.mj_step(
                self.model,
                self.data,
            )

        self.t += 1

        obs = self._get_obs()
        done = self.t >= self.horizon

        q_actual = self.data.qpos[
            self.qpos_ids
        ].copy()

        servo_error = (
                q_actual
                - q_target
        )

        info = {
            "tcp": self.tcp_feature().copy(),

            "qpos": self.data.qpos.copy().astype(
                np.float32
            ),

            "action": action.astype(
                np.float32
            ),

            "cartesian_delta": (
                requested_cartesian_delta.astype(
                    np.float32
                )
            ),

            "joint_delta": delta_q.astype(
                np.float32
            ),

            "q_target": q_target.astype(
                np.float32
            ),

            # IK计算后的理论末端位置残差
            "ik_position_error": float(
                ik_position_error
            ),

            # IK内部实际使用的迭代次数
            "ik_iterations": int(
                ik_iterations
            ),

            # 3×6位置雅可比的条件数
            "jacobian_condition": float(
                jacobian_condition
            ),

            # MuJoCo执行后实际关节角与目标关节角的差
            "servo_error": servo_error.astype(
                np.float32
            ),

            "q_reference": self.q_reference.astype(
                np.float32
            ),

            "tcp_reference": self.tcp_reference.astype(
                np.float32
            ),
        }

        return obs, 0.0, done, info

    def set_tracking_target(self, target_xyz: np.ndarray) -> np.ndarray:
        """设置下一次动作要跟踪的专家位置，并返回更新后的观测。"""
        self.tracking_target_xyz = np.asarray(
            target_xyz,
            dtype=np.float64,
        ).reshape(3).copy()
        return self._get_obs()

    def _site_velocity(self) -> np.ndarray:
        res = np.zeros(6, dtype=np.float64)
        try:
            self.mujoco.mj_objectVelocity(
                self.model,
                self.data,
                self.mujoco.mjtObj.mjOBJ_SITE,
                self.site_id,
                res,
                0,
            )
            angular = res[:3]
            linear = res[3:]
            return np.concatenate([linear, angular]).astype(np.float32)
        except Exception:
            return np.zeros(6, dtype=np.float32)

    def tcp_feature(self) -> np.ndarray:
        pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float32).reshape(3)
        rot = np.asarray(self.data.site_xmat[self.site_id], dtype=np.float32).reshape(3, 3)
        rotvec = rotmat_to_rotvec(rot)
        vel = self._site_velocity()
        return np.concatenate([pos, rotvec, vel]).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        qpos = np.asarray(
            self.data.qpos,
            dtype=np.float32,
        ).reshape(-1)

        qvel = np.asarray(
            self.data.qvel,
            dtype=np.float32,
        ).reshape(-1)

        tcp = self.tcp_feature()

        q_reference = (
            np.asarray(
                self.q_reference,
                dtype=np.float32,
            )
            if self.q_reference is not None
            else qpos[self.qpos_ids].copy()
        )

        tcp_reference_xyz = (
            np.asarray(
                self.tcp_reference[:3],
                dtype=np.float32,
            )
            if self.tcp_reference is not None
            else tcp[:3].copy()
        )

        reference_error = (
                tcp_reference_xyz
                - tcp[:3]
        )

        tracking_target_xyz = (
            np.asarray(
                self.tracking_target_xyz,
                dtype=np.float32,
            )
            if self.tracking_target_xyz is not None
            else tcp[:3].copy()
        )
        tracking_error = (
            tracking_target_xyz - tcp[:3]
        ) / self.tracking_error_scale

        phase = np.array(
            [
                min(
                    self.t
                    / max(1, self.horizon),
                    1.0,
                )
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                qpos,
                qvel,
                tcp,
                q_reference,
                tcp_reference_xyz,
                reference_error,
                tracking_target_xyz,
                tracking_error,
                phase,
            ]
        ).astype(np.float32)
