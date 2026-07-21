from __future__ import annotations

import numpy as np


def pairwise_sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)


def standardize_by_expert(traj: np.ndarray, expert: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = expert.mean(axis=0, keepdims=True)
    std = expert.std(axis=0, keepdims=True) + 1e-6
    return (traj - mean) / std, (expert - mean) / std


def sinkhorn_distance(
    traj: np.ndarray,
    expert: np.ndarray,
    epsilon: float = 0.1,
    n_iters: int = 100,
    standardize: bool = True,
) -> float:
    """
    计算两条轨迹之间的 Sinkhorn Distance。

    traj/expert: shape = [T, feature_dim]
    这里把每个时间点看成概率分布中的一个点，质量均匀分布。
    """
    x = np.asarray(traj, dtype=np.float64)
    y = np.asarray(expert, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("traj and expert must be 2-D arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature dim mismatch: {x.shape[1]} vs {y.shape[1]}")

    if standardize:
        x, y = standardize_by_expert(x, y)

    n, m = x.shape[0], y.shape[0]
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m

    cost = pairwise_sq_dist(x, y)
    # 防止 exp(-cost/epsilon) 下溢。
    eps = max(float(epsilon), 1e-6)
    K = np.exp(-cost / eps) + 1e-12

    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)
    for _ in range(int(n_iters)):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)

    transport = u[:, None] * K * v[None, :]
    return float(np.sum(transport * cost))
