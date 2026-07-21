from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F

from tgod_sd.configs import SACConfig
from tgod_sd.models.networks import DoubleQCritic, TanhGaussianActor
from tgod_sd.utils import to_device


class SACAgent:
    """
    标准 SAC 主体。

    这个类只负责 SAC 的三件事：
    1. 更新 Critic
    2. 更新 Actor
    3. 更新自动温度系数 alpha

    TGOD 的 MINE 伪奖励不放在这里，而是放在 tgod_sac.py。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        cfg: SACConfig,
        device: str = "auto",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.cfg = cfg
        self.device = to_device(device)

        self.actor = TanhGaussianActor(obs_dim, z_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic = DoubleQCritic(obs_dim, z_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim, z_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.log_alpha = torch.tensor(math.log(cfg.alpha_init), device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        self.target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(action_dim)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs, z, deterministic: bool = False):
        obs_t = torch.as_tensor(obs[None], device=self.device, dtype=torch.float32)
        z_t = torch.as_tensor(z[None], device=self.device, dtype=torch.float32)
        action, _ = self.actor.sample(obs_t, z_t, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def update_sac(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        action = batch["action"]
        z = batch["z"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        critic_loss = self._update_critic(obs, action, z, reward, next_obs, done)
        actor_loss, entropy = self._update_actor_and_alpha(obs, z)
        self._soft_update_targets()

        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha": float(self.alpha.detach().cpu()),
            "entropy": float(entropy),
        }

    def _update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs, z)
            target_q1, target_q2 = self.critic_target(next_obs, z, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            backup = reward + (1.0 - done) * self.cfg.gamma * target_q

        q1, q2 = self.critic(obs, z, action)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return float(critic_loss.detach().cpu())

    def _update_actor_and_alpha(self, obs: torch.Tensor, z: torch.Tensor) -> tuple[float, float]:
        new_action, log_prob = self.actor.sample(obs, z)
        q1_pi, q2_pi = self.critic(obs, z, new_action)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        entropy = -log_prob.detach().mean()
        return float(actor_loss.detach().cpu()), float(entropy.cpu())

    def _soft_update_targets(self) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(1.0 - tau)
                p_targ.data.add_(tau * p.data)

    def state_dict(self) -> dict[str, Any]:
        return {
            "sac_config": asdict(self.cfg),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])
