from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from tgod_sd.configs import EnvConfig
from tgod_sd.utils import rotmat_to_rotvec
from tgod_sd.xml_utils import prepare_runtime_xml


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
        self.action_scale = float(cfg.action_scale)
        self.reset_noise = float(cfg.reset_noise)
        self.t = 0
        self._runtime_tmp: Optional[tempfile.TemporaryDirectory] = None

        self.model, self.data = self._load_model(
            scene_xml=cfg.scene_xml,
            ur5e_xml=cfg.ur5e_xml,
            patch_mesh_assets=cfg.patch_mesh_assets,
        )

        self.nu = int(self.model.nu)
        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.qpos_ids = np.arange(min(self.nu, self.nq), dtype=np.int64)

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

        # 清除上一个episode的时间、力、加速度、
        # 接触和执行器内部状态
        self.mujoco.mj_resetData(
            self.model,
            self.data,
        )

        # 每一个episode都使用完全相同的专家起点
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel

        if self.nu > 0:
            q = self.data.qpos[
                self.qpos_ids[: self.nu]
            ]
            self.data.ctrl[:] = np.clip(
                q,
                self.ctrl_low,
                self.ctrl_high,
            )

        self.mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(self.nu)
        action = np.clip(action, -1.0, 1.0)

        q = self.data.qpos[self.qpos_ids[: self.nu]].copy()
        q_target = q + self.action_scale * action
        self.data.ctrl[:] = np.clip(q_target, self.ctrl_low, self.ctrl_high)

        for _ in range(self.frame_skip):
            self.mujoco.mj_step(self.model, self.data)

        self.t += 1
        obs = self._get_obs()
        done = self.t >= self.horizon
        env_reward = 0.0
        info = {
            "tcp": self.tcp_feature().copy(),
            "qpos": self.data.qpos.copy().astype(np.float32),
            "action": action.astype(np.float32),
        }
        return obs, env_reward, done, info

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
        # qpos = np.asarray(self.data.qpos, dtype=np.float32).reshape(-1)
        # qvel = np.asarray(self.data.qvel, dtype=np.float32).reshape(-1)
        # tcp = self.tcp_feature()
        # return np.concatenate([qpos, qvel, tcp]).astype(np.float32)
        qpos = np.asarray(
            self.data.qpos,
            dtype=np.float32,
        ).reshape(-1)

        qvel = np.asarray(
            self.data.qvel,
            dtype=np.float32,
        ).reshape(-1)

        tcp = self.tcp_feature()

        phase = np.array(
            [
                self.t
                / max(1, self.horizon - 1)
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [qpos, qvel, tcp, phase]
        ).astype(np.float32)
