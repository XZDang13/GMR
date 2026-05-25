import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


CANONICAL_BIND_OFFSET_ENCODING = "RootWorldAndLocalRotationsWithBindOffsets"
OVR_FULL_BODY_MAJOR_JOINTS_SOURCE = "OVRFullBodyMajorJoints"
OVR_FULL_BODY_CANONICAL_SOURCE = "OVRFullBodyCanonicalBindPose"
UNITY_WORLD_COORDINATE_SPACE = "UnityWorld"
ROOT_POSITION_MODES = ("live", "zero-horizontal", "freeze-horizontal")

# Unity: x right, y up, z forward.
# GMR/MuJoCo: x forward, y left, z up.
UNITY_TO_GMR_MATRIX = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

G1_23DOF_REQUIRED_JOINTS = (
    "Hips",
    "Chest",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
    "LeftToes",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "RightToes",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftLowerArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
)


@dataclass(frozen=True)
class IobtReplayData:
    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]]
    actual_human_height: Optional[float]
    fps: float
    metadata: Dict[str, Any]


def infer_iobt_src_human(metadata: Dict[str, Any]) -> str:
    if is_ovr_full_body_metadata(metadata):
        return "iobt_ovr_fullbody"
    return "iobt_mocopi"


def is_ovr_full_body_metadata(metadata: Dict[str, Any]) -> bool:
    return metadata.get("source") in (
        OVR_FULL_BODY_MAJOR_JOINTS_SOURCE,
        OVR_FULL_BODY_CANONICAL_SOURCE,
    )


def load_iobt_replay_file(
    replay_file: str,
    *,
    zero_horizontal_root: Optional[bool] = None,
    root_position_mode: str = "zero-horizontal",
    start: Optional[int] = None,
    end: Optional[int] = None,
    required_joint_names: Optional[Sequence[str]] = G1_23DOF_REQUIRED_JOINTS,
) -> IobtReplayData:
    if zero_horizontal_root is not None:
        root_position_mode = "zero-horizontal" if zero_horizontal_root else "live"
    if root_position_mode not in ROOT_POSITION_MODES:
        raise ValueError(
            f"root_position_mode must be one of {', '.join(ROOT_POSITION_MODES)}; "
            f"got {root_position_mode!r}"
        )

    metadata, frame_events = read_iobt_replay_events(replay_file)
    validate_iobt_metadata(metadata, required_joint_names)

    selected_events = frame_events[slice(start, end)]
    frames = build_gmr_human_frames(metadata, selected_events)
    frames = apply_root_position_mode(frames, root_position_mode)

    return IobtReplayData(
        frames=frames,
        actual_human_height=metadata_height(metadata),
        fps=infer_replay_fps(selected_events),
        metadata=metadata,
    )


def apply_root_position_mode(
    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    root_position_mode: str,
) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    if root_position_mode == "live" or not frames:
        return frames

    adjusted_frames = []
    first_root = frames[0]["Hips"][0]
    first_horizontal = np.array([first_root[0], first_root[1], 0.0], dtype=float)
    for frame in frames:
        if root_position_mode == "zero-horizontal":
            horizontal_offset = first_horizontal
        else:
            root = frame["Hips"][0]
            horizontal_offset = np.array([root[0], root[1], 0.0], dtype=float)

        adjusted_frames.append(
            {
                body_name: (position - horizontal_offset, rotation)
                for body_name, (position, rotation) in frame.items()
            }
        )

    return adjusted_frames


def read_iobt_replay_events(replay_file: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    path = Path(replay_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    metadata: Optional[Dict[str, Any]] = None
    frame_events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid replay JSON") from exc

            event_type = event.get("type")
            if event_type == "metadata":
                metadata = event.get("metadata")
            elif event_type == "frame":
                frame_events.append(event)

    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: replay does not contain a metadata event")
    if not frame_events:
        raise ValueError(f"{path}: replay does not contain frame events")

    return metadata, frame_events


def validate_iobt_metadata(
    metadata: Dict[str, Any],
    required_joint_names: Optional[Sequence[str]] = G1_23DOF_REQUIRED_JOINTS,
) -> None:
    pose_encoding = metadata.get("poseEncoding")
    coordinate_space = metadata.get("coordinateSpace")
    source = metadata.get("source")
    is_canonical = pose_encoding == CANONICAL_BIND_OFFSET_ENCODING
    is_ovr_unity_world = (
        source == OVR_FULL_BODY_MAJOR_JOINTS_SOURCE
        and coordinate_space == UNITY_WORLD_COORDINATE_SPACE
        and not pose_encoding
    )
    if not is_canonical and not is_ovr_unity_world:
        raise ValueError(
            "IOBT replay must either use canonical poseEncoding "
            f"{CANONICAL_BIND_OFFSET_ENCODING!r} or source "
            f"{OVR_FULL_BODY_MAJOR_JOINTS_SOURCE!r} in {UNITY_WORLD_COORDINATE_SPACE!r}; "
            f"got source={source!r}, coordinateSpace={coordinate_space!r}, poseEncoding={pose_encoding!r}"
        )

    joints = metadata.get("joints")
    if not isinstance(joints, list):
        raise ValueError("IOBT replay metadata must contain a joints list")

    if required_joint_names is not None:
        available = {
            str(joint.get("name"))
            for joint in joints
            if isinstance(joint, dict) and joint.get("name")
        }
        missing = sorted(set(required_joint_names) - available)
        if missing:
            raise ValueError("IOBT replay metadata is missing required joints: " + ", ".join(missing))


def build_gmr_human_frames(
    metadata: Dict[str, Any],
    frame_events: Iterable[Dict[str, Any]],
) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    joint_defs = metadata.get("joints") or []
    frames: List[Dict[str, Tuple[np.ndarray, np.ndarray]]] = []
    for frame_event in frame_events:
        unity_world = build_unity_world_pose(metadata, joint_defs, frame_event)
        frame = {
            name: (
                unity_position_to_gmr(position),
                unity_quat_xyzw_to_gmr_wxyz(rotation_xyzw),
            )
            for name, position, rotation_xyzw in unity_world
        }
        if metadata.get("source") == OVR_FULL_BODY_CANONICAL_SOURCE:
            frame = apply_ovr_semantic_root_rotation(frame)
        frames.append(frame)

    return frames


def apply_ovr_semantic_root_rotation(
    frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    root_rotation = ovr_semantic_root_rotation(frame)
    if root_rotation is None:
        return frame

    updated = dict(frame)
    root_position, _raw_root_rotation = updated["Hips"]
    updated["Hips"] = (root_position, root_rotation.as_quat(scalar_first=True))
    return updated


def ovr_semantic_root_rotation(
    frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Optional[R]:
    required = ("Hips", "Chest", "LeftUpperLeg", "RightUpperLeg")
    if any(name not in frame for name in required):
        return None

    left_hip = np.asarray(frame["LeftUpperLeg"][0], dtype=float)
    right_hip = np.asarray(frame["RightUpperLeg"][0], dtype=float)
    lateral = left_hip - right_hip
    lateral[2] = 0.0
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm < 1e-6:
        return None
    lateral = lateral / lateral_norm

    up = np.array([0.0, 0.0, 1.0], dtype=float)
    forward = np.cross(lateral, up)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return None
    forward = forward / forward_norm
    lateral = np.cross(up, forward)
    lateral = lateral / np.linalg.norm(lateral)

    return R.from_matrix(np.column_stack([forward, lateral, up]))


def build_unity_world_pose(
    metadata: Dict[str, Any],
    joint_defs: Sequence[Dict[str, Any]],
    frame_event: Dict[str, Any],
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    frame_joints = frame_event.get("joints")
    if not isinstance(frame_joints, list):
        raise ValueError("IOBT frame event does not contain a joints list")
    if len(frame_joints) < len(joint_defs):
        raise ValueError(
            f"IOBT frame has {len(frame_joints)} joints but metadata defines {len(joint_defs)}"
        )

    if is_unity_world_metadata(metadata):
        return build_legacy_unity_world_pose(joint_defs, frame_joints)

    return build_canonical_unity_world_pose(joint_defs, frame_joints)


def is_unity_world_metadata(metadata: Dict[str, Any]) -> bool:
    # Version-1 UnityWorld replays store live world pose directly in each frame record.
    # Canonical bind-offset replays store root world pose plus child local rotations.
    return (
        metadata.get("source") == OVR_FULL_BODY_MAJOR_JOINTS_SOURCE
        and metadata.get("coordinateSpace") == UNITY_WORLD_COORDINATE_SPACE
        and not metadata.get("poseEncoding")
    )


def build_legacy_unity_world_pose(
    joint_defs: Sequence[Dict[str, Any]],
    frame_joints: Sequence[Dict[str, Any]],
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    world_pose: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for index, joint_def in enumerate(joint_defs):
        name = str(joint_def.get("name") or f"joint_{index}")
        frame_joint = frame_joints[index]
        if not isinstance(frame_joint, dict):
            raise ValueError(f"IOBT frame joint {index} is not an object")

        position = vector3_from_value(frame_joint.get("position"))
        rotation = normalize_quat_xyzw(rotation_from_xyzw(frame_joint.get("rotation")).as_quat())
        world_pose.append((name, position, rotation))

    return world_pose


def build_canonical_unity_world_pose(
    joint_defs: Sequence[Dict[str, Any]],
    frame_joints: Sequence[Dict[str, Any]],
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    world_positions: List[np.ndarray] = []
    world_rotations: List[R] = []
    world_pose: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for index, joint_def in enumerate(joint_defs):
        name = str(joint_def.get("name") or f"joint_{index}")
        frame_joint = frame_joints[index]
        if not isinstance(frame_joint, dict):
            raise ValueError(f"IOBT frame joint {index} is not an object")

        rotation_value = frame_joint.get("rotation")
        if rotation_value is None:
            rotation_value = joint_def.get("bindLocalRotation")
        local_rotation = rotation_from_xyzw(rotation_value)

        parent_index = int(joint_def.get("parentIndex", -1))
        if parent_index < 0:
            position = vector3_from_value(frame_joint.get("position"))
            rotation = local_rotation
        else:
            if parent_index >= len(world_positions):
                raise ValueError(f"Joint {name} parent index {parent_index} is not available")

            bind_offset = vector3_from_value(joint_def.get("bindLocalPosition"))
            parent_rotation = world_rotations[parent_index]
            position = world_positions[parent_index] + parent_rotation.apply(bind_offset)
            rotation = parent_rotation * local_rotation

        rotation_xyzw = normalize_quat_xyzw(rotation.as_quat())
        world_positions.append(position)
        world_rotations.append(R.from_quat(rotation_xyzw))
        world_pose.append((name, position, rotation_xyzw))

    return world_pose


def unity_position_to_gmr(position: Sequence[float]) -> np.ndarray:
    return UNITY_TO_GMR_MATRIX @ np.asarray(position, dtype=float)


def unity_quat_xyzw_to_gmr_wxyz(quat_xyzw: Sequence[float]) -> np.ndarray:
    unity_rotation = rotation_from_xyzw(quat_xyzw)
    gmr_matrix = UNITY_TO_GMR_MATRIX @ unity_rotation.as_matrix() @ UNITY_TO_GMR_MATRIX.T
    quat_xyzw_out = normalize_quat_xyzw(R.from_matrix(gmr_matrix).as_quat())
    return quat_xyzw_out[[3, 0, 1, 2]]


def vector3_from_value(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        return np.array(
            [
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            ],
            dtype=float,
        )
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return np.asarray(value[:3], dtype=float)
    raise ValueError(f"Expected Vector3-like value, got {value!r}")


def rotation_from_xyzw(value: Any) -> R:
    if isinstance(value, dict):
        try:
            quat = np.array(
                [
                    float(value.get("x", 0.0)),
                    float(value.get("y", 0.0)),
                    float(value.get("z", 0.0)),
                    float(value.get("w", 1.0)),
                ],
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected Quaternion xyzw value, got {value!r}") from exc
        return R.from_quat(normalize_quat_xyzw(quat))

    try:
        quat = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected Quaternion xyzw value, got {value!r}") from exc
    if quat.ndim == 0 or quat.shape[0] < 4:
        raise ValueError(f"Expected Quaternion xyzw value, got {value!r}")
    return R.from_quat(normalize_quat_xyzw(quat[:4]))


def normalize_quat_xyzw(value: Sequence[float]) -> np.ndarray:
    quat = np.asarray(value, dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return quat / norm


def infer_replay_fps(frame_events: Sequence[Dict[str, Any]], default_fps: float = 30.0) -> float:
    timestamps = [
        float(event["timestamp"])
        for event in frame_events
        if isinstance(event.get("timestamp"), (int, float))
    ]
    if len(timestamps) < 2:
        return default_fps

    deltas = np.diff(np.asarray(timestamps, dtype=float))
    deltas = deltas[np.isfinite(deltas) & (deltas > 1e-6)]
    if deltas.size == 0:
        return default_fps

    return float(1.0 / np.median(deltas))


def metadata_height(metadata: Dict[str, Any]) -> Optional[float]:
    value = metadata.get("skeletonHeightMeters")
    try:
        height = float(value)
    except (TypeError, ValueError):
        return None
    return height if 0.25 < height < 3.0 else None
