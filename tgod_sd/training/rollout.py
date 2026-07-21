from __future__ import annotations

import numpy as np

from tgod_sd.algorithms.replay_buffer import ReplayBuffer
from tgod_sd.algorithms.tgod_sac import TGODSACAgent
from tgod_sd.envs.ur5e_env import UR5eTGODEnv


def rollout_episode(
    env: UR5eTGODEnv,
    agent: TGODSACAgent,
    replay: ReplayBuffer | None = None,
    start_steps_left: int = 0,
    deterministic: bool = False,
    fixed_z: bool = False,
) -> dict[str, np.ndarray | float | int]:
    """跑一个 episode。训练时传 replay；评估/匹配时 replay=None。"""
    obs = env.reset()
    if fixed_z:
        z = np.zeros(
            agent.z_dim,
            dtype=np.float32,
        )
    else:
        z = agent.sample_z(1)[0]
    initial_obs = obs.copy()
    initial_tcp = env.tcp_feature().copy()
    initial_qpos = env.data.qpos.copy().astype(
        np.float32
    )
    initial_qvel = env.data.qvel.copy().astype(
        np.float32
    )

    tcp_list: list[np.ndarray] = []
    qpos_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    reward_list: list[float] = []

    done = False
    steps = 0
    while not done:
        if replay is not None and start_steps_left > 0:
            action = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            start_steps_left -= 1
        else:
            action = agent.act(obs, z, deterministic=deterministic).astype(np.float32)

        # reward = agent.compute_pseudo_reward(obs, z)
        next_obs, _, done, info = env.step(action)
        reward = agent.compute_tracking_reward(
            tcp=info["tcp"],
            step=env.t,
            action=action,
        )

        if replay is not None:
            replay.add(obs, action, z, reward, next_obs, done)

        tcp_list.append(info["tcp"])
        qpos_list.append(info["qpos"])
        action_list.append(action)
        reward_list.append(float(reward))

        obs = next_obs
        steps += 1

    return {
        "z": z.astype(np.float32),

        # 动作执行前的真正初始状态
        "initial_obs": initial_obs.astype(np.float32),
        "initial_tcp": initial_tcp.astype(np.float32),
        "initial_qpos": initial_qpos,
        "initial_qvel": initial_qvel,

        # 每一步动作执行后的状态
        "tcp": np.asarray(
            tcp_list,
            dtype=np.float32,
        ),
        "qpos": np.asarray(
            qpos_list,
            dtype=np.float32,
        ),
        "action": np.asarray(
            action_list,
            dtype=np.float32,
        ),

        "reward_sum": float(
            np.sum(reward_list)
        ),
        "reward_mean": (
            float(np.mean(reward_list))
            if reward_list
            else 0.0
        ),
        "steps": steps,
        "start_steps_left": start_steps_left,
    }
