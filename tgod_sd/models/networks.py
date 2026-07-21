from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for _ in range(depth):
        layers += [nn.Linear(last, hidden_dim), nn.ReLU()]
        last = hidden_dim
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    """SAC Actor：输入 [s,z]，输出 tanh-squashed Gaussian 动作。"""

    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = build_mlp(obs_dim + z_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([obs, z], dim=-1))
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean, log_std = self.forward(obs, z)
        if deterministic:
            action = torch.tanh(mean)
            return action, None

        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)

        # tanh 高斯策略的 log_prob 修正项。
        log_prob = normal.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class DoubleQCritic(nn.Module):
    """SAC Critic：双 Q 网络，用于缓解 Q 过估计。"""

    def __init__(self, obs_dim: int, z_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + z_dim + action_dim
        self.q1 = build_mlp(input_dim, hidden_dim, 1)
        self.q2 = build_mlp(input_dim, hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, z, action], dim=-1)
        return self.q1(x), self.q2(x)


class MINENet(nn.Module):
    """
    MINE 互信息估计器。

    对应 TGOD 里的两个互信息项：
    - mine_zs: I(Z;S)
    - mine_zd: I(Z;D)
    """

    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = build_mlp(x_dim + z_dim, hidden_dim, 1)

    def score(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=-1))

    def mi_lower_bound(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        joint = self.score(x, z)
        z_shuffle = z[torch.randperm(z.shape[0], device=z.device)]
        marginal = self.score(x, z_shuffle)
        # Donsker-Varadhan lower bound: E[T_joint] - log E[exp(T_marginal)]
        return joint.mean() - torch.logsumexp(marginal, dim=0).mean() + torch.log(
            torch.tensor(float(marginal.shape[0]), device=x.device)
        )
