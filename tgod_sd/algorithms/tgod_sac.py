from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from tgod_sd.algorithms.replay_buffer import ReplayBuffer
from tgod_sd.algorithms.sac import SACAgent
from tgod_sd.configs import SACConfig, TGODConfig
from tgod_sd.models.networks import MINENet
from tgod_sd.utils import rotation_error_rotvec


class TGODSACAgent(SACAgent):
    """
    TGOD + SAC。

    与标准 SAC 的区别：
    - 额外维护两个 MINE 网络：mine_zs, mine_zd
    - 每一步交互时计算 TGOD 伪奖励：R = MINE(z;s) + MINE(z;D)
    - 更新时先更新 MINE，再调用 SAC 的 update_sac()
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            expert_demo: np.ndarray,
            tcp_slice: slice,
            sac_cfg: SACConfig,
            tgod_cfg: TGODConfig,
            device: str = "auto",
    ):
        super().__init__(obs_dim, action_dim, tgod_cfg.z_dim, sac_cfg, device=device)
        self.tgod_cfg = tgod_cfg
        self.tcp_slice = tcp_slice

        self.mine_zs = MINENet(13, tgod_cfg.z_dim, sac_cfg.hidden_dim).to(self.device)
        # D只有一条固定示范时，I(Z;D)本身退化为0。这里使用论文
        # “围绕示范生成”的可计算形式：当前TCP、同相位专家TCP和残差。
        self.mine_zd = MINENet(9, tgod_cfg.z_dim, sac_cfg.hidden_dim).to(self.device)
        self.mine_opt = torch.optim.Adam(
            list(self.mine_zs.parameters()) + list(self.mine_zd.parameters()),
            lr=tgod_cfg.lr_mine,
        )

        self.expert_demo = expert_demo.astype(np.float32)
        self.expert_tensor = torch.as_tensor(self.expert_demo, device=self.device)
        self.expert_mean = self.expert_demo.mean(axis=0, keepdims=True)
        self.expert_std = self.expert_demo.std(axis=0, keepdims=True) + 1e-6

    def sample_z(self, batch: int = 1) -> np.ndarray:
        return np.random.randn(batch, self.z_dim).astype(np.float32)

    def update(
            self,
            replay: ReplayBuffer,
            batch_size: int,
    ) -> dict[str, float]:
        if replay.size < batch_size:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "mine_loss": 0.0,
                "mi_zs": 0.0,
                "mi_zd": 0.0,
                "alpha": float(
                    self.alpha.detach().cpu()
                ),
                "entropy": 0.0,
            }

        batch = replay.sample(
            batch_size,
            self.device,
        )

        mode = self.tgod_cfg.reward_mode

        if mode == "tcp_tracking":
            sac_stats = self.update_sac(batch)

            return {
                **sac_stats,
                "mine_loss": 0.0,
                "mi_zs": 0.0,
                "mi_zd": 0.0,
            }

        if mode == "tgod":
            mine_stats = self._update_mine(
                batch["next_obs"],
                batch["z"],
            )

            with torch.no_grad():
                pseudo_reward, reward_stats = (
                    self._pseudo_reward_batch(
                        batch["next_obs"],
                        batch["z"],
                    )
                )
            batch["reward"] = pseudo_reward

            sac_stats = self.update_sac(
                batch
            )

            return {
                **sac_stats,
                **mine_stats,
                **reward_stats,
            }

        raise ValueError(
            f"未知reward_mode：{mode}"
        )

    def _update_mine(self, obs: torch.Tensor, z: torch.Tensor) -> dict[str, float]:
        mi_zs = self.mine_zs.mi_lower_bound(
            self._state_condition(obs),
            z,
        )
        demo_condition = self._demo_condition(obs)
        mi_zd = self.mine_zd.mi_lower_bound(
            demo_condition,
            z,
        )
        mine_loss = -(
            self.tgod_cfg.mine_zs_weight * mi_zs
            + self.tgod_cfg.mine_zd_weight * mi_zd
        )

        self.mine_opt.zero_grad(set_to_none=True)
        mine_loss.backward()
        self.mine_opt.step()

        return {
            "mine_loss": float(mine_loss.detach().cpu()),
            "mi_zs": float(mi_zs.detach().cpu()),
            "mi_zd": float(mi_zd.detach().cpu()),
        }

    @torch.no_grad()
    def compute_pseudo_reward(self, obs: np.ndarray, z: np.ndarray) -> float:
        obs_t = torch.as_tensor(obs[None], device=self.device, dtype=torch.float32)
        z_t = torch.as_tensor(z[None], device=self.device, dtype=torch.float32)
        demo_condition = self._demo_condition(obs_t)
        r_tgod = (
            self.tgod_cfg.mine_zs_weight
            * self.mine_zs.score(
                self._state_condition(obs_t),
                z_t,
            )
            + self.tgod_cfg.mine_zd_weight
            * self.mine_zd.score(demo_condition, z_t)
        )
        r = r_tgod.item()

        if self.tgod_cfg.anchor_reward_weight > 0:
            tcp = obs[self.tcp_slice][None, :]
            r += self.tgod_cfg.anchor_reward_weight * float(self._anchor_reward_np(tcp)[0])

        r *= self.tgod_cfg.reward_scale
        return float(np.clip(r, -self.tgod_cfg.reward_clip, self.tgod_cfg.reward_clip))

    def _demo_condition(self, obs: torch.Tensor) -> torch.Tensor:
        tcp_xyz = obs[:, self.tcp_slice.start:self.tcp_slice.start + 3]
        phase = torch.clamp(obs[:, -1], 0.0, 1.0)
        index = torch.round(
            phase * float(len(self.expert_demo) - 1)
        ).long()
        expert_xyz = self.expert_tensor[index, :3]
        mean = torch.as_tensor(
            self.expert_mean[0, :3],
            device=self.device,
        )
        std = torch.as_tensor(
            self.expert_std[0, :3],
            device=self.device,
        )
        current_norm = (tcp_xyz - mean) / std
        expert_norm = (expert_xyz - mean) / std
        error_norm = (
            tcp_xyz - expert_xyz
        ) / max(float(self.tgod_cfg.position_error_scale), 1e-6)
        return torch.cat(
            [current_norm, expert_norm, error_norm],
            dim=-1,
        )

    def _state_condition(self, obs: torch.Tensor) -> torch.Tensor:
        tcp = obs[:, self.tcp_slice]
        mean = torch.as_tensor(
            self.expert_mean[0],
            device=self.device,
        )
        std = torch.as_tensor(
            np.maximum(self.expert_std[0], 1e-3),
            device=self.device,
        )
        tcp_norm = torch.clamp(
            (tcp - mean) / std,
            -10.0,
            10.0,
        )
        return torch.cat(
            [tcp_norm, obs[:, -1:]],
            dim=-1,
        )

    def _pseudo_reward_batch(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """用当前MINE计算逐样本DV密度比，避免经验池中的旧伪奖励。"""
        demo_condition = self._demo_condition(obs)
        shuffled_z = z[torch.randperm(z.shape[0], device=z.device)]

        state_condition = self._state_condition(obs)
        joint_zs = self.mine_zs.score(state_condition, z)
        marginal_zs = self.mine_zs.score(
            state_condition,
            shuffled_z,
        )
        joint_zd = self.mine_zd.score(demo_condition, z)
        marginal_zd = self.mine_zd.score(
            demo_condition,
            shuffled_z,
        )

        log_partition_zs = torch.logsumexp(
            marginal_zs,
            dim=0,
            keepdim=True,
        ) - np.log(float(z.shape[0]))
        log_partition_zd = torch.logsumexp(
            marginal_zd,
            dim=0,
            keepdim=True,
        ) - np.log(float(z.shape[0]))

        reward_zs = joint_zs - log_partition_zs
        reward_zd = joint_zd - log_partition_zd
        reward = self.tgod_cfg.reward_scale * (
            self.tgod_cfg.mine_zs_weight * reward_zs
            + self.tgod_cfg.mine_zd_weight * reward_zd
        )
        reward = torch.clamp(
            reward,
            -self.tgod_cfg.reward_clip,
            self.tgod_cfg.reward_clip,
        )
        stats = {
            "pseudo_reward_mean": float(reward.mean().cpu()),
            "pseudo_reward_std": float(reward.std(unbiased=False).cpu()),
            "reward_zs_mean": float(reward_zs.mean().cpu()),
            "reward_zd_mean": float(reward_zd.mean().cpu()),
        }
        return reward, stats

    def _sample_expert_torch(self, batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, self.expert_tensor.shape[0], (batch_size,), device=self.device)
        return self.expert_tensor[idx]

    def _anchor_reward_np(self, tcp_batch: np.ndarray) -> np.ndarray:
        """
        可选稳定项：到专家 TCP 轨迹最近点的负距离。
        注意：这不是论文核心 TGOD 公式，只建议前期调通时使用。
        """
        x = (tcp_batch[:, :12] - self.expert_mean) / self.expert_std
        y = (self.expert_demo - self.expert_mean) / self.expert_std
        out: list[float] = []
        for xb in x:
            d2 = np.sum((y - xb[None, :]) ** 2, axis=1)
            out.append(-float(np.sqrt(np.min(d2) + 1e-9)))
        return np.asarray(out, dtype=np.float32)

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "tgod_config": asdict(self.tgod_cfg),
                "mine_zs": self.mine_zs.state_dict(),
                "mine_zd": self.mine_zd.state_dict(),
                "mine_opt": self.mine_opt.state_dict(),
                "expert_mean": self.expert_mean,
                "expert_std": self.expert_std,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.mine_zs.load_state_dict(state["mine_zs"])
        self.mine_zd.load_state_dict(state["mine_zd"])
        self.mine_opt.load_state_dict(state["mine_opt"])

    def compute_tracking_reward(
            self,
            tcp: np.ndarray,
            step: int,
            action: np.ndarray,
    ) -> float:
        index = min(
            int(step),
            len(self.expert_demo) - 1,
        )

        target = self.expert_demo[index]

        position_error = (
                tcp[:3] - target[:3]
        )

        action_cost = float(
            np.sum(np.square(action))
        )

        reward = (
                -50.0
                * float(
            np.sum(
                np.square(position_error)
            )
        )
                - 0.001 * action_cost
        )

        return float(
            np.clip(reward, -20.0, 5.0)
        )

    def compute_joint_tracking_reward(
            self,
            qpos: np.ndarray,
            action: np.ndarray,
            step: int,
            expert_qpos: np.ndarray,
    ) -> float:
        index = min(
            int(step),
            len(expert_qpos) - 1,
        )

        q_target = expert_qpos[index]

        raw_error = qpos - q_target

        # 把关节角误差限制到 [-pi, pi]
        q_error = np.arctan2(
            np.sin(raw_error),
            np.cos(raw_error),
        )

        normalized_error = q_error / np.pi

        tracking_cost = float(
            np.mean(normalized_error ** 2)
        )

        action_cost = float(
            np.mean(action ** 2)
        )

        reward = (
                -10.0 * tracking_cost
                - 0.001 * action_cost
        )

        return float(
            np.clip(reward, -20.0, 1.0)
        )

    def compute_tcp_tracking_reward(
        self,
        tcp: np.ndarray,
        step: int,
        action: np.ndarray,
    ) -> float:
        index = min(
            max(int(step), 0),
            len(self.expert_demo) - 1,
        )

        current_position = np.asarray(
            tcp[:3],
            dtype=np.float32,
        )

        target_position = self.expert_demo[
            index,
            :3,
        ]

        position_distance = float(
            np.linalg.norm(
                current_position
                - target_position
            )
        )

        action_cost = float(
            np.mean(
                np.asarray(
                    action,
                    dtype=np.float32,
                ) ** 2
            )
        )

        position_scale = max(
            float(self.tgod_cfg.position_error_scale),
            1e-6,
        )

        # 使用可配置的无量纲距离。5 cm误差在默认配置下对应-1，
        # 同时保留很小的动作正则项。
        reward = (
            -self.tgod_cfg.position_reward_weight
            * position_distance
            / position_scale
            -self.tgod_cfg.action_penalty_weight
            * action_cost
        )

        return float(
            np.clip(
                reward,
                -self.tgod_cfg.reward_clip,
                0.0,
            )
        )
