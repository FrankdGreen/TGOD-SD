from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from tgod_sd.configs import EnvConfig
from tgod_sd.envs.ur5e_env import UR5eTGODEnv
from tgod_sd.utils import load_expert_demo


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation vector -> 3x3 rotation matrix."""
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = v / theta
    x, y, z = axis
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)



def rotmat_to_rotvec_stable(rotmat: np.ndarray) -> np.ndarray:
    """Numerically stable 3x3 rotation matrix -> rotation vector."""
    R = np.asarray(rotmat, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-10:
        return np.zeros(3, dtype=np.float64)

    # The usual formula is ill-conditioned around pi.
    if np.pi - theta < 1e-5:
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        index = int(np.argmax(axis))
        if axis[index] < 1e-8:
            return np.array([theta, 0.0, 0.0], dtype=np.float64)
        if index == 0:
            axis[1] = A[0, 1] / axis[0]
            axis[2] = A[0, 2] / axis[0]
        elif index == 1:
            axis[0] = A[0, 1] / axis[1]
            axis[2] = A[1, 2] / axis[1]
        else:
            axis[0] = A[0, 2] / axis[2]
            axis[1] = A[1, 2] / axis[2]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return axis / norm * theta

    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * np.sin(theta))
    return axis * theta


def clamp_joint_limits(env: UR5eTGODEnv, qpos: np.ndarray) -> None:
    """Clamp only joints that are explicitly limited in MJCF."""
    model = env.model
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        low, high = model.jnt_range[joint_id]
        qpos[qpos_adr] = np.clip(qpos[qpos_adr], low, high)


def solve_ik_point(
    env: UR5eTGODEnv,
    target_tcp: np.ndarray,
    q_seed: np.ndarray,
    *,
    max_iters: int,
    damping: float,
    orientation_weight: float,
    position_tolerance: float,
    orientation_tolerance: float,
    max_joint_step: float,
) -> tuple[np.ndarray, float, float, int]:
    """Damped least-squares IK for one TCP pose, seeded by the previous point."""
    target = np.asarray(target_tcp, dtype=np.float64).reshape(-1)
    if target.shape[0] < 6:
        raise ValueError("TCP target must contain at least [x,y,z,rx,ry,rz].")

    target_pos = target[:3]
    target_rot = rotvec_to_rotmat(target[3:6])
    q = np.asarray(q_seed, dtype=np.float64).copy()

    jac_pos = np.zeros((3, env.nv), dtype=np.float64)
    jac_rot = np.zeros((3, env.nv), dtype=np.float64)
    pos_norm = np.inf
    rot_norm = np.inf

    for iteration in range(max_iters):
        env.data.qpos[:] = q
        env.data.qvel[:] = 0.0
        env.mujoco.mj_forward(env.model, env.data)

        current_pos = np.asarray(env.data.site_xpos[env.site_id], dtype=np.float64).copy()
        current_rot = np.asarray(env.data.site_xmat[env.site_id], dtype=np.float64).reshape(3, 3).copy()

        pos_error = target_pos - current_pos
        # World-frame orientation error; compatible with mj_jacSite rotational Jacobian.
        rot_error = rotmat_to_rotvec_stable(target_rot @ current_rot.T)
        pos_norm = float(np.linalg.norm(pos_error))
        rot_norm = float(np.linalg.norm(rot_error))

        if pos_norm <= position_tolerance and rot_norm <= orientation_tolerance:
            return q, pos_norm, rot_norm, iteration

        env.mujoco.mj_jacSite(
            env.model,
            env.data,
            jac_pos,
            jac_rot,
            env.site_id,
        )

        weight = float(orientation_weight)
        jacobian = np.vstack([jac_pos, weight * jac_rot])
        error = np.concatenate([pos_error, weight * rot_error])
        regularized = jacobian @ jacobian.T + (damping**2) * np.eye(6)
        try:
            dq = jacobian.T @ np.linalg.solve(regularized, error)
        except np.linalg.LinAlgError:
            dq = np.linalg.pinv(jacobian, rcond=1e-5) @ error

        largest = float(np.max(np.abs(dq))) if dq.size else 0.0
        if largest > max_joint_step:
            dq *= max_joint_step / largest
        q += 0.7 * dq
        clamp_joint_limits(env, q)

    return q, pos_norm, rot_norm, max_iters


def build_expert_qpos(
    env: UR5eTGODEnv,
    expert_tcp: np.ndarray,
    cache_path: Path,
    *,
    rebuild: bool,
    max_iters: int,
    damping: float,
    orientation_weight: float,
) -> np.ndarray:
    """Convert the expert TCP trajectory into a continuous joint trajectory."""
    if cache_path.exists() and not rebuild:
        cached = np.asarray(np.load(cache_path), dtype=np.float64)
        if cached.ndim == 2 and cached.shape == (len(expert_tcp), env.nq):
            print(f"[IK] 使用缓存的专家关节轨迹：{cache_path}")
            return cached
        print(f"[IK] 缓存尺寸不匹配，将重新计算：{cached.shape}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    q_seed = env.home_qpos.copy()
    q_trajectory: list[np.ndarray] = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    failed_indices: list[int] = []

    print(f"[IK] 开始将 {len(expert_tcp)} 个 TCP 点转换为关节角……")
    for index, target in enumerate(expert_tcp):
        q, pos_error, rot_error, iterations = solve_ik_point(
            env,
            target,
            q_seed,
            max_iters=max_iters,
            damping=damping,
            orientation_weight=orientation_weight,
            position_tolerance=1e-5,
            orientation_tolerance=1e-4,
            max_joint_step=0.15,
        )
        q_trajectory.append(q.copy())
        position_errors.append(pos_error)
        orientation_errors.append(rot_error)
        q_seed = q

        if iterations >= max_iters:
            failed_indices.append(index)
        if (index + 1) % 100 == 0 or index + 1 == len(expert_tcp):
            print(
                f"[IK] {index + 1:4d}/{len(expert_tcp)} "
                f"位置误差={pos_error:.3e} m，姿态误差={rot_error:.3e} rad"
            )

    result = np.asarray(q_trajectory, dtype=np.float64)
    np.save(cache_path, result.astype(np.float32))
    print(f"[IK] 已保存专家关节轨迹：{cache_path}")
    print(
        f"[IK] 平均位置误差={np.mean(position_errors):.3e} m，"
        f"最大位置误差={np.max(position_errors):.3e} m"
    )
    print(
        f"[IK] 平均姿态误差={np.mean(orientation_errors):.3e} rad，"
        f"最大姿态误差={np.max(orientation_errors):.3e} rad"
    )
    if failed_indices:
        print(f"[WARN] 有 {len(failed_indices)} 个点达到迭代上限：{failed_indices[:20]}")
    return result


def load_best_trajectory(path: Path, nq: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到最优轨迹文件：{path}\n"
            "请把 --best_file 改成实际训练输出，例如 "
            "outputs_test/best_tgod_sd_trajectory.npz。"
        )
    with np.load(path, allow_pickle=False) as data:
        if "qpos" not in data:
            raise KeyError(f"{path} 中没有 qpos 字段，可用字段：{data.files}")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] < nq:
            raise ValueError(f"qpos 尺寸应为 (N, >= {nq})，实际为 {qpos.shape}")
        qpos = qpos[:, :nq]
        if "tcp" in data:
            tcp = np.asarray(data["tcp"], dtype=np.float64)
        else:
            tcp = np.empty((len(qpos), 0), dtype=np.float64)
    return qpos, tcp


def set_kinematic_state(env: UR5eTGODEnv, qpos: np.ndarray) -> None:
    env.data.qpos[:] = qpos[: env.nq]
    env.data.qvel[:] = 0.0
    if env.nu > 0:
        env.data.ctrl[:] = np.clip(qpos[: env.nu], env.ctrl_low, env.ctrl_high)
    env.mujoco.mj_forward(env.model, env.data)


def set_dynamic_target(env: UR5eTGODEnv, qpos: np.ndarray, steps_per_frame: int) -> None:
    env.data.ctrl[:] = np.clip(qpos[: env.nu], env.ctrl_low, env.ctrl_high)
    for _ in range(max(1, steps_per_frame)):
        env.mujoco.mj_step(env.model, env.data)


def add_sphere(
    mujoco_module,
    viewer,
    position: np.ndarray,
    radius: float,
    rgba: Iterable[float],
) -> bool:
    index = int(viewer.user_scn.ngeom)
    if index >= len(viewer.user_scn.geoms):
        return False
    mujoco_module.mjv_initGeom(
        viewer.user_scn.geoms[index],
        type=mujoco_module.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0.0, 0.0], dtype=np.float64),
        pos=np.asarray(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).reshape(-1),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom = index + 1
    return True


def draw_paths(
    env: UR5eTGODEnv,
    viewer,
    expert_xyz: np.ndarray | None,
    generated_xyz: np.ndarray | None,
    current_xyz: np.ndarray,
    stride: int,
) -> None:
    if not hasattr(viewer, "user_scn"):
        return


    viewer.user_scn.ngeom = 0
    stride = max(1, int(stride))

    # Expert: red; generated: blue; current TCP: yellow.
    if expert_xyz is not None:
        for point in expert_xyz[::stride]:
            if not add_sphere(env.mujoco, viewer, point, 0.0045, [0.95, 0.15, 0.10, 0.85]):
                break
    if generated_xyz is not None:
        for point in generated_xyz[::stride]:
            if not add_sphere(env.mujoco, viewer, point, 0.0045, [0.10, 0.35, 0.95, 0.85]):
                break
    add_sphere(env.mujoco, viewer, current_xyz, 0.012, [1.0, 0.85, 0.05, 1.0])


def configure_camera(viewer, xyz_sets: list[np.ndarray]) -> None:
    valid = [x for x in xyz_sets if x is not None and x.size > 0]
    if valid:
        points = np.concatenate(valid, axis=0)
        center = np.mean(points, axis=0)
        span = float(np.max(np.ptp(points, axis=0)))
    else:
        center = np.array([0.0, 0.0, 0.4], dtype=np.float64)
        span = 1.0
    viewer.cam.lookat[:] = center
    viewer.cam.distance = max(1.2, 2.2 * span)
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -22.0


def replay_sequence(
    env: UR5eTGODEnv,
    viewer,
    name: str,
    q_trajectory: np.ndarray,
    *,
    expert_xyz: np.ndarray | None,
    generated_xyz: np.ndarray | None,
    playback_mode: str,
    fps: float,
    speed: float,
    steps_per_frame: int,
    trail_stride: int,
    state: dict[str, bool],
) -> bool:
    print(f"[PLAY] 正在播放：{name}，共 {len(q_trajectory)} 帧")
    frame_period = 1.0 / max(1e-6, fps * speed)
    frame = 0
    state["restart"] = False

    # Start exactly at the first state in either mode.
    with viewer.lock():
        set_kinematic_state(env, q_trajectory[0])
        current_xyz = np.asarray(env.data.site_xpos[env.site_id], dtype=np.float64).copy()
        draw_paths(env, viewer, expert_xyz, generated_xyz, current_xyz, trail_stride)
    viewer.sync()

    while viewer.is_running() and frame < len(q_trajectory):
        if state["restart"]:
            frame = 0
            state["restart"] = False
        if state["paused"]:
            viewer.sync()
            time.sleep(0.03)
            continue

        start = time.perf_counter()
        with viewer.lock():
            if playback_mode == "kinematic":
                set_kinematic_state(env, q_trajectory[frame])
            else:
                set_dynamic_target(env, q_trajectory[frame], steps_per_frame)
            current_xyz = np.asarray(env.data.site_xpos[env.site_id], dtype=np.float64).copy()
            draw_paths(env, viewer, expert_xyz, generated_xyz, current_xyz, trail_stride)
        viewer.sync()

        frame += 1
        remaining = frame_period - (time.perf_counter() - start)
        if remaining > 0:
            time.sleep(remaining)

    return viewer.is_running()


def wait_with_viewer(viewer, seconds: float, state: dict[str, bool]) -> None:
    end = time.perf_counter() + max(0.0, seconds)
    while viewer.is_running() and time.perf_counter() < end:
        viewer.sync()
        time.sleep(0.03)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay expert TCP demonstration and TGOD-SD best trajectory in MuJoCo."
    )
    parser.add_argument("--mode", choices=["expert", "best", "both"], default="both")
    parser.add_argument("--expert_demo", default="data/expert_demo.npy")
    parser.add_argument("--best_file", default="outputs/best_tgod_sd_trajectory.npz")
    parser.add_argument("--expert_qpos_cache", default="outputs_replay/expert_qpos.npy")
    parser.add_argument(
        "--scene_xml",
        default="data/scene.xml",
        help="The visual model is preferred because its assets directory is included.",
    )
    parser.add_argument("--ur5e_xml", default="data/ur5e.xml")
    parser.add_argument("--patch_mesh_assets", choices=["auto", "always", "never"], default="never")
    parser.add_argument("--site_name", default="attachment_site")
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 is real-time; 0.5 is half speed.")
    parser.add_argument("--playback_mode", choices=["kinematic", "dynamic"], default="kinematic")
    parser.add_argument("--steps_per_frame", type=int, default=5)
    parser.add_argument("--trail_stride", type=int, default=5)
    parser.add_argument("--hold", type=float, default=1.5)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--prepare_only", action="store_true", help="Only compute/check trajectories; do not open GUI.")
    parser.add_argument("--rebuild_ik", action="store_true")
    parser.add_argument("--ik_max_iters", type=int, default=100)
    parser.add_argument("--ik_damping", type=float, default=1e-3)
    parser.add_argument("--orientation_weight", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.speed <= 0:
        raise ValueError("--fps and --speed must be positive.")

    env_cfg = EnvConfig(
        scene_xml=args.scene_xml,
        ur5e_xml=args.ur5e_xml,
        horizon=500,
        frame_skip=args.steps_per_frame,
        action_scale=0.05,
        site_name=args.site_name,
        patch_mesh_assets=args.patch_mesh_assets,
        reset_noise=0.0,
    )
    env = UR5eTGODEnv(env_cfg)

    expert_tcp = load_expert_demo(args.expert_demo)
    expert_xyz = np.asarray(expert_tcp[:, :3], dtype=np.float64)
    expert_qpos: np.ndarray | None = None
    best_qpos: np.ndarray | None = None
    generated_xyz: np.ndarray | None = None

    try:
        if args.mode in {"expert", "both"}:
            expert_qpos = build_expert_qpos(
                env,
                expert_tcp,
                Path(args.expert_qpos_cache),
                rebuild=args.rebuild_ik,
                max_iters=args.ik_max_iters,
                damping=args.ik_damping,
                orientation_weight=args.orientation_weight,
            )

        if args.mode in {"best", "both"}:
            best_qpos, best_tcp = load_best_trajectory(Path(args.best_file), env.nq)
            if best_tcp.ndim == 2 and best_tcp.shape[1] >= 3:
                generated_xyz = best_tcp[:, :3]
            else:
                generated_xyz = None
            print(f"[LOAD] 最优关节轨迹：{best_qpos.shape}，文件：{args.best_file}")

        if args.prepare_only:
            print("[DONE] 轨迹准备和尺寸检查完成，未打开 MuJoCo 窗口。")
            return 0

        try:
            import mujoco.viewer  # type: ignore
        except Exception as exc:
            raise RuntimeError("无法导入 mujoco.viewer，请确认已执行 pip install mujoco。") from exc

        state = {"paused": False, "restart": False}

        def key_callback(keycode: int) -> None:
            try:
                key = chr(keycode).lower()
            except (ValueError, OverflowError):
                return
            if key == " ":
                state["paused"] = not state["paused"]
                print("[KEY] 暂停" if state["paused"] else "[KEY] 继续")
            elif key == "r":
                state["restart"] = True
                state["paused"] = False
                print("[KEY] 从当前序列第一帧重新播放")

        sequences: list[tuple[str, np.ndarray]] = []
        if expert_qpos is not None:
            sequences.append(("专家示范（TCP经逆运动学转换）", expert_qpos))
        if best_qpos is not None:
            sequences.append(("TGOD-SD生成的最优轨迹", best_qpos))

        print("[VIEWER] 空格：暂停/继续；R：重播当前序列；关闭窗口：结束。")

        with mujoco.viewer.launch_passive(
                env.model,
                env.data,
                key_callback=key_callback,
        ) as viewer:
            with viewer.lock():
                configure_camera(
                    viewer,
                    [expert_xyz, generated_xyz]
                    if generated_xyz is not None
                    else [expert_xyz],
                )

            viewer.sync()

            while viewer.is_running():
                for name, trajectory in sequences:
                    if not replay_sequence(
                            env,
                            viewer,
                            name,
                            trajectory,
                            expert_xyz=expert_xyz,
                            generated_xyz=generated_xyz,
                            playback_mode=args.playback_mode,
                            fps=args.fps,
                            speed=args.speed,
                            steps_per_frame=args.steps_per_frame,
                            trail_stride=args.trail_stride,
                            state=state,
                    ):
                        break

                    wait_with_viewer(
                        viewer,
                        args.hold,
                        state,
                    )

                if not args.loop or not viewer.is_running():
                    break

        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] 用户中止。")
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
