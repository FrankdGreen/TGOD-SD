from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tgod_sd.algorithms.replay_buffer import ReplayBuffer
from tgod_sd.algorithms.tgod_sac import TGODSACAgent
from tgod_sd.configs import EnvConfig, MatchConfig, SACConfig, TGODConfig, TrainConfig
from tgod_sd.envs.ur5e_env import UR5eTGODEnv
from tgod_sd.matching.sinkhorn import sinkhorn_distance
from tgod_sd.training.rollout import rollout_episode
from tgod_sd.utils import ensure_dir, json_dumps, load_expert_demo, set_seed
from tgod_sd.visualization.plotting import plot_expert_vs_generated


class TGODSDTrainer:
    """训练组织器：只负责把 Env、ReplayBuffer、Agent、Matcher 串起来。"""

    def __init__(
            self,
            env_cfg: EnvConfig,
            sac_cfg: SACConfig,
            tgod_cfg: TGODConfig,
            train_cfg: TrainConfig,
            match_cfg: MatchConfig,
    ):
        self.env_cfg = env_cfg
        self.sac_cfg = sac_cfg
        self.tgod_cfg = tgod_cfg
        self.train_cfg = train_cfg
        self.match_cfg = match_cfg

        set_seed(train_cfg.seed)
        self.output_dir = ensure_dir(train_cfg.output_dir)
        self.expert_demo = load_expert_demo(train_cfg.expert_demo)

        self.env = UR5eTGODEnv(env_cfg)
        self._validate_initial_state(),
        self.agent = TGODSACAgent(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            expert_demo=self.expert_demo,
            tcp_slice=self.env.tcp_slice,
            sac_cfg=sac_cfg,
            tgod_cfg=tgod_cfg,
            device=train_cfg.device,
        )
        self.replay = ReplayBuffer(
            capacity=train_cfg.replay_size,
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            z_dim=tgod_cfg.z_dim,
        )

    def train(self) -> None:
        self._save_config()
        log_path = self.output_dir / "train_log.jsonl"
        start_steps_left = self.train_cfg.start_steps

        with log_path.open("w", encoding="utf-8") as f:
            for ep in range(1, self.train_cfg.episodes + 1):
                episode = rollout_episode(
                    self.env,
                    self.agent,
                    replay=self.replay,
                    start_steps_left=start_steps_left,
                    deterministic=False,
                    fixed_z=self.train_cfg.fixed_z,
                )
                start_steps_left = int(episode["start_steps_left"])

                stats = {}

                if self.replay.size >= self.train_cfg.learning_starts:
                    update_times = (
                            int(episode["steps"])
                            * self.train_cfg.updates_per_step
                    )

                    for _ in range(update_times):
                        stats = self.agent.update(
                            self.replay,
                            self.train_cfg.batch_size,
                        )

                record = {
                    "episode": ep,
                    "reward_sum": episode["reward_sum"],
                    "reward_mean": episode["reward_mean"],
                    "replay_size": self.replay.size,
                    "start_steps_left": start_steps_left,
                    **stats,
                }
                f.write(json_dumps(record) + "\n")
                f.flush()

                if ep % self.train_cfg.log_interval == 0 or ep == 1:
                    print(
                        f"[EP {ep:04d}] "
                        f"R={record['reward_sum']:.2f} "
                        f"critic={record.get('critic_loss', 0):.3f} "
                        f"actor={record.get('actor_loss', 0):.3f} "
                        f"mine={record.get('mine_loss', 0):.3f} "
                        f"alpha={record.get('alpha', 0):.3f}"
                    )

                if ep % self.train_cfg.save_interval == 0:
                    self.save_checkpoint(self.output_dir / "tgod_sd_checkpoint.pt")
                    np.save(self.output_dir / "last_episode_tcp.npy", episode["tcp"])

        self.save_checkpoint(self.output_dir / "tgod_sd_checkpoint.pt")
        best = self.match_best_trajectory()
        self._save_best(best)

    def match_best_trajectory(self) -> dict:
        best: dict | None = None
        for i in range(self.match_cfg.n_candidates):
            episode = rollout_episode(
                self.env,
                self.agent,
                replay=None,
                deterministic=self.match_cfg.deterministic,
                fixed_z=self.train_cfg.fixed_z,
            )
            tcp, qpos = self._aligned_trajectory(
                episode
            )
            dist = sinkhorn_distance(
                tcp,
                self.expert_demo,
                epsilon=self.match_cfg.sinkhorn_epsilon,
                n_iters=self.match_cfg.sinkhorn_iters,
                standardize=True,
            )
            print(f"[MATCH] candidate {i + 1}/{self.match_cfg.n_candidates}, SD={dist:.6f}")
            if best is None or dist < best["sinkhorn_distance"]:
                best = {
                    "sinkhorn_distance": dist,
                    "tcp": tcp,
                    "qpos": qpos,
                    "initial_tcp": episode["initial_tcp"],
                    "initial_qpos": episode["initial_qpos"],
                    "initial_qvel": episode["initial_qvel"],
                    "action": episode["action"],
                    "z": episode["z"],
                    "reward_sum": episode["reward_sum"],
                    "cartesian_delta": episode[
                        "cartesian_delta"
                    ],

                    "joint_delta": episode[
                        "joint_delta"
                    ],

                    "q_target": episode[
                        "q_target"
                    ],

                    "jacobian_condition": episode[
                        "jacobian_condition"
                    ],
                }
        assert best is not None
        return best

    def _save_best(self, best: dict) -> None:
        out_npz = self.output_dir / "best_tgod_sd_trajectory.npz"
        np.savez(
            out_npz,
            tcp=best["tcp"],
            qpos=best["qpos"],
            action=best["action"],
            z=best["z"],
            initial_tcp=best["initial_tcp"],
            initial_qpos=best["initial_qpos"],
            initial_qvel=best["initial_qvel"],
            sinkhorn_distance=np.asarray(
                best["sinkhorn_distance"],
                dtype=np.float32,
            ),
            expert_demo=self.expert_demo,
            cartesian_delta=best[
                "cartesian_delta"
            ],

            joint_delta=best[
                "joint_delta"
            ],

            q_target=best[
                "q_target"
            ],

            jacobian_condition=best[
                "jacobian_condition"
            ],
        )
        plot_expert_vs_generated(
            expert=self.expert_demo,
            generated=best["tcp"],
            output_path=self.output_dir / "expert_vs_tgod_sd_tcp.png",
        )
        print(f"[DONE] best trajectory saved to: {out_npz}")

    def _save_config(self) -> None:
        cfg = {
            "env": asdict(self.env_cfg),
            "sac": asdict(self.sac_cfg),
            "tgod": asdict(self.tgod_cfg),
            "train": asdict(self.train_cfg),
            "match": asdict(self.match_cfg),
            "expert_shape": list(self.expert_demo.shape),
        }
        (self.output_dir / "config.json").write_text(json_dumps(cfg), encoding="utf-8")

    def save_checkpoint(self, path: str | Path) -> None:
        state = {
            "agent": self.agent.state_dict(),
            "env_config": asdict(self.env_cfg),
            "sac_config": asdict(self.sac_cfg),
            "tgod_config": asdict(self.tgod_cfg),
            "train_config": asdict(self.train_cfg),
            "match_config": asdict(self.match_cfg),
        }
        torch.save(state, path)

    def close(self) -> None:
        self.env.close()

    def _validate_initial_state(self) -> None:
        self.env.reset()

        generated_tcp0 = self.env.tcp_feature()
        expert_tcp0 = self.expert_demo[0]

        position_error = float(
            np.linalg.norm(
                generated_tcp0[:3]
                - expert_tcp0[:3]
            )
        )

        orientation_error = float(
            np.linalg.norm(
                generated_tcp0[3:6]
                - expert_tcp0[3:6]
            )
        )

        linear_velocity_error = float(
            np.linalg.norm(
                generated_tcp0[6:9]
                - expert_tcp0[6:9]
            )
        )

        angular_velocity_error = float(
            np.linalg.norm(
                generated_tcp0[9:12]
                - expert_tcp0[9:12]
            )
        )

        print("[ALIGN] 环境初始TCP：", generated_tcp0)
        print("[ALIGN] 专家初始TCP：", expert_tcp0)
        print(
            f"[ALIGN] 位置误差："
            f"{position_error:.6e} m"
        )
        print(
            f"[ALIGN] 姿态误差："
            f"{orientation_error:.6e} rad"
        )
        print(
            f"[ALIGN] 线速度误差："
            f"{linear_velocity_error:.6e} m/s"
        )
        print(
            f"[ALIGN] 角速度误差："
            f"{angular_velocity_error:.6e} rad/s"
        )

        if position_error > 1e-3:
            raise RuntimeError(
                "环境起点与专家起点的位置误差超过1 mm，"
                "请检查IK、模型或坐标系。"
            )

    def _aligned_trajectory(
            self,
            episode: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        专家有500帧，其中第0帧是初始状态。

        当前环境执行500个动作会得到500个动作后状态。
        为保持输出仍为500帧，使用：
        初始状态 + 前499个动作后状态。
        """
        tcp_after = np.asarray(
            episode["tcp"],
            dtype=np.float32,
        )
        qpos_after = np.asarray(
            episode["qpos"],
            dtype=np.float32,
        )

        tcp_aligned = np.concatenate(
            [
                np.asarray(
                    episode["initial_tcp"],
                    dtype=np.float32,
                )[None, :],
                tcp_after[:-1],
            ],
            axis=0,
        )

        qpos_aligned = np.concatenate(
            [
                np.asarray(
                    episode["initial_qpos"],
                    dtype=np.float32,
                )[None, :],
                qpos_after[:-1],
            ],
            axis=0,
        )

        target_length = len(
            self.expert_demo
        )

        return (
            tcp_aligned[:target_length],
            qpos_aligned[:target_length],
        )
