from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def load_expert_demo(path: str | Path) -> np.ndarray:
    """
    加载专家示范。

    当前工程默认使用 12 维 TCP 特征：
    [x, y, z, rx, ry, rz, vx, vy, vz, wx, wy, wz]
    如果用户给的维度少于 12，会补零；多于 12，只取前 12 维。
    """
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        obj = arr.item() if arr.shape == () else arr
        if isinstance(obj, dict):
            for key in ["expert_demo", "trajectory", "traj", "demo", "data"]:
                if key in obj:
                    arr = obj[key]
                    break

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expert demo must be 2-D, got shape={arr.shape}")
    if arr.shape[1] < 6:
        raise ValueError("Expert demo must contain at least [x,y,z,rx,ry,rz].")
    if arr.shape[1] < 12:
        pad = np.zeros((arr.shape[0], 12 - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr[:, :6], pad], axis=1)
    elif arr.shape[1] > 12:
        arr = arr[:, :12]
    return arr.astype(np.float32)


def rotmat_to_rotvec(rotmat_3x3: np.ndarray) -> np.ndarray:
    """不依赖 scipy 的旋转矩阵 -> 旋转向量转换。"""
    R = np.asarray(rotmat_3x3, dtype=np.float64).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)

    denom = max(2.0 * np.sin(theta), 1e-12)
    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    ) / denom
    return (axis * theta).astype(np.float32)


def to_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def skew(
        vector: np.ndarray,
) -> np.ndarray:
    x, y, z = np.asarray(
        vector,
        dtype=np.float64,
    )

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def rotvec_to_rotmat(
        rotvec: np.ndarray,
) -> np.ndarray:
    rotvec = np.asarray(
        rotvec,
        dtype=np.float64,
    ).reshape(3)

    theta = float(
        np.linalg.norm(rotvec)
    )

    if theta < 1e-10:
        return (
                np.eye(3)
                + skew(rotvec)
        )

    axis = rotvec / theta
    axis_skew = skew(axis)

    return (
            np.eye(3)
            + np.sin(theta) * axis_skew
            + (
                    1.0 - np.cos(theta)
            )
            * axis_skew
            @ axis_skew
    )


def rotation_error_rotvec(
        current_rotvec: np.ndarray,
        target_rotvec: np.ndarray,
) -> np.ndarray:
    current_rotation = rotvec_to_rotmat(
        current_rotvec
    )

    target_rotation = rotvec_to_rotmat(
        target_rotvec
    )

    # 世界坐标系中的目标相对当前旋转
    error_rotation = (
            target_rotation
            @ current_rotation.T
    )

    return rotmat_to_rotvec(
        error_rotation
    )
