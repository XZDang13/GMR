import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer


UNITY_TO_GMR = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
GMR_TO_UNITY = UNITY_TO_GMR.T
SMPLX_ROTATION_FRAME = "SMPL-X world joint frame"
# Pelvis stays identity: the raw Quest/Unity hips frame is already close to the
# SMPL-X root frame. The legacy -90deg X correction made the robot lean backward.
SMPLX_GMR_LOCAL_FRAME_CORRECTIONS = {
    "pelvis": R.identity(),
    "spine2": R.from_quat([1.0, 0.0, 0.0, 0.0]),
    "left_collar": R.from_quat([0.0, 0.0, 0.7071067812, 0.7071067812]),
    "left_shoulder": R.from_quat([0.0, 0.0, 0.7071067812, 0.7071067812]),
    "left_elbow": R.from_quat([0.0, 0.0, 0.7071067812, 0.7071067812]),
    "left_wrist": R.from_quat([-0.5, -0.5, 0.5, 0.5]),
    "right_collar": R.from_quat([0.0, 0.0, 0.7071067812, 0.7071067812]),
    "right_shoulder": R.from_quat([0.0, 0.0, 0.7071067812, 0.7071067812]),
    "right_elbow": R.from_quat([0.0, 0.0, 0.7071067812, 0.7071067812]),
    "right_wrist": R.from_quat([0.7071067812, 0.7071067812, 0.0, 0.0]),
    "left_hip": R.from_quat([1.0, 0.0, 0.0, 0.0]),
    "left_knee": R.from_quat([1.0, 0.0, 0.0, 0.0]),
    "left_ankle": R.from_quat([0.7071067812, 0.0, 0.0, 0.7071067812]),
    "left_foot": R.from_quat([0.7071067812, 0.0, 0.0, 0.7071067812]),
    "right_ankle": R.from_quat([-0.7071067812, 0.0, 0.0, 0.7071067812]),
    "right_foot": R.from_quat([-0.7071067812, 0.0, 0.0, 0.7071067812]),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget IOBT SMPL-X body-motion JSONL through GMR's SMPL-X source config."
    )
    parser.add_argument("--input", required=True, help="IOBT smplx_motion JSONL file.")
    parser.add_argument("--robot", choices=("unitree_g1_23dof",), default="unitree_g1_23dof")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/iobt_smplx_g1_23dof.mp4")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--damping", type=float, default=5e-1)
    parser.add_argument("--ik-mode", choices=("adaptive", "single-pass"), default="adaptive")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Disable orientation IK costs; useful when source joint axes are not SMPL-X bind axes.",
    )
    parser.add_argument(
        "--identity-rotations",
        action="store_true",
        help="Use SMPL-X neutral world orientations for all joints.",
    )
    parser.add_argument(
        "--disable-legacy-frame-correction",
        action="store_true",
        help="Do not repair legacy Unity/OVR bone rotations into SMPL-X joint frames.",
    )
    parser.add_argument(
        "--calibrate-initial-rotations",
        action="store_true",
        help=(
            "Remove each joint's first recorded world orientation. Off by default because "
            "the first motion frame is not a bind pose."
        ),
    )
    parser.add_argument(
        "--no-calibrate-initial-rotations",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--root-origin",
        action="store_true",
        help="Subtract the first pelvis XY position after converting to GMR coordinates.",
    )
    args = parser.parse_args()

    frames = load_iobt_smplx_frames(
        Path(args.input).expanduser(),
        start=args.start,
        end=args.end,
        max_frames=args.max_frames,
        identity_rotations=args.identity_rotations,
        repair_legacy_rotation_frame=not args.disable_legacy_frame_correction,
        calibrate_initial_rotations=(
            args.calibrate_initial_rotations and not args.no_calibrate_initial_rotations
        ),
        root_origin=args.root_origin,
    )
    if not frames:
        raise ValueError("No smplx_motion_frame records were loaded.")

    fps = max(1, int(round(args.fps if args.fps is not None else infer_fps(frames))))
    human_height = args.height if args.height is not None else estimate_human_height(frames)

    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=human_height,
        solver=args.solver,
        damping=args.damping,
        ik_mode=args.ik_mode,
        max_iter=args.max_iter,
        verbose=args.verbose,
    )
    if args.position_only:
        disable_orientation_costs(retargeter)

    viewer = None
    if not args.no_viewer:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=fps,
            camera_follow=False,
            record_video=args.record_video,
            video_path=args.video_path,
        )

    qpos_list: List[np.ndarray] = []
    progress = tqdm(total=len(frames), desc="Retargeting IOBT SMPL-X motion")
    try:
        for frame in frames:
            qpos = retargeter.retarget(frame["human_data"])
            qpos_list.append(qpos.copy())
            if viewer is not None:
                viewer.step(
                    qpos[:3],
                    qpos[3:7],
                    qpos[7:],
                    human_motion_data=retargeter.scaled_human_data,
                    rate_limit=args.rate_limit,
                    follow_camera=False,
                )
            progress.update(1)
    except KeyboardInterrupt:
        pass
    finally:
        progress.close()
        if viewer is not None:
            viewer.close()

    if args.save_path and qpos_list:
        save_robot_motion(args.save_path, fps, qpos_list)
        print(f"Saved {len(qpos_list)} frames to {args.save_path}")


def load_iobt_smplx_frames(
    path: Path,
    start: Optional[int],
    end: Optional[int],
    max_frames: Optional[int],
    identity_rotations: bool,
    repair_legacy_rotation_frame: bool,
    calibrate_initial_rotations: bool,
    root_origin: bool,
) -> List[Dict[str, Any]]:
    raw_frames = list(_iter_raw_motion_frames(path, start, end, max_frames))
    if not raw_frames:
        return []

    initial_rotations = first_joint_rotations(raw_frames[0]) if calibrate_initial_rotations else {}
    root_xy = np.zeros(2, dtype=float)
    if root_origin:
        root_pos = frame_pelvis_position(raw_frames[0])
        root_xy = (UNITY_TO_GMR @ root_pos)[:2]

    frames = []
    for raw in raw_frames:
        human_data = {}
        for joint in raw.get("joints", []):
            name = joint.get("smplxName")
            if not name:
                continue
            position = UNITY_TO_GMR @ vector3(joint.get("worldPosition", [0.0, 0.0, 0.0]))
            position[:2] -= root_xy

            if identity_rotations:
                quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                rotation = unity_rotation_to_gmr(joint.get("worldRotationQuatXyzw", [0.0, 0.0, 0.0, 1.0]))
                if repair_legacy_rotation_frame and not uses_smplx_rotation_frame(raw, joint):
                    rotation = repair_legacy_smplx_rotation(name, rotation)
                initial = initial_rotations.get(name)
                if initial is not None:
                    rotation = rotation * initial.inv()
                quat_wxyz = rotation.as_quat(scalar_first=True)

            human_data[name] = [position, normalize_quat_wxyz(quat_wxyz)]

        frames.append(
            {
                "human_data": human_data,
                "receivedAtUnix": raw.get("receivedAtUnix"),
                "timestamp": raw.get("timestamp"),
                "sequence": raw.get("sequence"),
            }
        )
    return frames


def _iter_raw_motion_frames(
    path: Path,
    start: Optional[int],
    end: Optional[int],
    max_frames: Optional[int],
) -> Iterable[Dict[str, Any]]:
    kept = 0
    frame_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if event.get("type") != "smplx_motion_frame":
                continue
            if end is not None and frame_index >= end:
                return
            if start is None or frame_index >= start:
                yield event
                kept += 1
                if max_frames is not None and kept >= max_frames:
                    return
            frame_index += 1


def first_joint_rotations(frame: Dict[str, Any]) -> Dict[str, R]:
    rotations = {}
    for joint in frame.get("joints", []):
        name = joint.get("smplxName")
        if name:
            rotations[name] = unity_rotation_to_gmr(
                joint.get("worldRotationQuatXyzw", [0.0, 0.0, 0.0, 1.0])
            )
    return rotations


def frame_pelvis_position(frame: Dict[str, Any]) -> np.ndarray:
    for joint in frame.get("joints", []):
        if joint.get("smplxName") == "pelvis":
            return vector3(joint.get("worldPosition", [0.0, 0.0, 0.0]))
    return vector3(frame.get("transl", [0.0, 0.0, 0.0]))


def unity_rotation_to_gmr(rotation_xyzw: Sequence[float]) -> R:
    unity_rotation = R.from_quat(quat_xyzw(rotation_xyzw))
    matrix = UNITY_TO_GMR @ unity_rotation.as_matrix() @ GMR_TO_UNITY
    return R.from_matrix(matrix)


def uses_smplx_rotation_frame(frame: Dict[str, Any], joint: Dict[str, Any]) -> bool:
    return (
        joint.get("rotationFrame") == SMPLX_ROTATION_FRAME
        or frame.get("rotationFrame") == SMPLX_ROTATION_FRAME
    )


def repair_legacy_smplx_rotation(name: str, rotation: R) -> R:
    correction = SMPLX_GMR_LOCAL_FRAME_CORRECTIONS.get(name)
    if correction is None:
        return rotation
    return rotation * correction


def vector3(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected vector shape (3,), got {array.shape}")
    return array


def quat_xyzw(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {array.shape}")
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return array / norm


def normalize_quat_wxyz(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return array / norm


def infer_fps(frames: List[Dict[str, Any]], default_fps: float = 30.0) -> float:
    for key in ("receivedAtUnix", "timestamp"):
        times = [float(frame[key]) for frame in frames if frame.get(key) is not None]
        if len(times) < 2:
            continue
        deltas = np.diff(np.asarray(times, dtype=float))
        deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
        if len(deltas) > 0:
            return float(1.0 / np.median(deltas))
    return default_fps


def estimate_human_height(frames: List[Dict[str, Any]], default_height: float = 1.66) -> float:
    heights = []
    for frame in frames[:: max(1, len(frames) // 120)]:
        positions = [value[0] for value in frame["human_data"].values()]
        if positions:
            z_values = np.asarray([position[2] for position in positions], dtype=float)
            height = float(np.max(z_values) - np.min(z_values))
            if np.isfinite(height) and height > 0.5:
                heights.append(height)
    if not heights:
        return default_height
    return float(np.median(heights))


def disable_orientation_costs(retargeter: GeneralMotionRetargeting) -> None:
    for task in list(getattr(retargeter, "tasks1", [])) + list(getattr(retargeter, "tasks2", [])):
        set_orientation_cost = getattr(task, "set_orientation_cost", None)
        if callable(set_orientation_cost):
            set_orientation_cost(0.0)


def save_robot_motion(save_path: str, fps: int, qpos_list: List[np.ndarray]) -> None:
    path = Path(save_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    qpos = np.asarray(qpos_list, dtype=float)
    motion_data = {
        "fps": fps,
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    with path.open("wb") as handle:
        pickle.dump(motion_data, handle)


if __name__ == "__main__":
    main()
