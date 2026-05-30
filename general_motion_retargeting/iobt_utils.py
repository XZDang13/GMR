import copy
import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


IOBT_CANONICAL_SRC_HUMAN = "iobt_canonical"
CANONICAL_BIND_OFFSET_ENCODING = "RootWorldAndLocalRotationsWithBindOffsets"
DEFAULT_CANONICAL_HUMAN_HEIGHT = 1.750136137
CANONICAL_HEIGHT_SOURCE_OVERRIDES = {
    "SMPLXBodyOnlyBindPose": DEFAULT_CANONICAL_HUMAN_HEIGHT,
}

CANONICAL_JOINT_NAMES = (
    "Hips",
    "Spine",
    "Chest",
    "UpperChest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftLowerArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
    "LeftToes",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "RightToes",
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
    "LeftUpperArm",
    "LeftLowerArm",
    "LeftHand",
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
)

UNITY_TO_GMR = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
GMR_TO_UNITY = np.linalg.inv(UNITY_TO_GMR)

HAND_FORWARD_LOCAL_AXES = {
    "Left": np.array([0.0, 1.0, 0.0]),
    "Right": np.array([0.0, -1.0, 0.0]),
}
HAND_FORWARD_LOCAL_AXIS = np.array([1.0, 0.0, 0.0])
HAND_PALM_NORMAL_LOCAL_AXES = {
    "Left": np.array([1.0, 0.0, 0.0]),
    "Right": np.array([1.0, 0.0, 0.0]),
}
HAND_ROLL_LOCAL_CORRECTION_DEGREES = {
    "Left": -90.0,
    "Right": -90.0,
}


@dataclass
class IOBTSkeletonFrame:
    human_data: Dict[str, List[np.ndarray]]
    sequence: Optional[int] = None
    timestamp: Optional[float] = None
    received_at_unix: Optional[float] = None


def copy_human_data(human_data: Dict[str, Sequence[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    return {
        name: [np.asarray(value[0], dtype=float).copy(), np.asarray(value[1], dtype=float).copy()]
        for name, value in human_data.items()
    }


def unity_position_to_gmr(position: Sequence[float]) -> np.ndarray:
    return UNITY_TO_GMR @ np.asarray(position, dtype=float)


def unity_rotation_to_gmr(rotation_xyzw: Sequence[float]) -> np.ndarray:
    unity_rotation = _rotation_from_xyzw(rotation_xyzw)
    gmr_matrix = UNITY_TO_GMR @ unity_rotation.as_matrix() @ GMR_TO_UNITY
    return R.from_matrix(gmr_matrix).as_quat(scalar_first=True)


def validate_iobt_metadata(
    metadata: Dict[str, Any],
    required_joint_names: Sequence[str] = G1_23DOF_REQUIRED_JOINTS,
) -> None:
    if metadata.get("poseEncoding") != CANONICAL_BIND_OFFSET_ENCODING:
        raise ValueError(
            "IOBT skeleton metadata must use poseEncoding "
            f"{CANONICAL_BIND_OFFSET_ENCODING!r}"
        )

    joint_defs = metadata.get("joints")
    if not isinstance(joint_defs, list) or not joint_defs:
        raise ValueError("IOBT skeleton metadata must contain a non-empty joints list")

    names = [joint.get("name") for joint in joint_defs if isinstance(joint, dict)]
    missing = [name for name in required_joint_names if name not in names]
    if missing:
        raise ValueError("IOBT skeleton metadata is missing required joints: " + ", ".join(missing))

    for index, joint in enumerate(joint_defs):
        if not isinstance(joint, dict):
            raise ValueError(f"IOBT skeleton joint definition {index} is not an object")
        parent_index = int(joint.get("parentIndex", -1))
        if parent_index >= index:
            raise ValueError(
                f"IOBT skeleton joint {joint.get('name', index)!r} must appear after its parent"
            )
        if parent_index < -1:
            raise ValueError(f"IOBT skeleton joint {joint.get('name', index)!r} has invalid parentIndex")
        _vector3(joint.get("bindLocalPosition", [0.0, 0.0, 0.0]))
        _quat_xyzw(joint.get("bindLocalRotation", [0.0, 0.0, 0.0, 1.0]))


def infer_iobt_fps(frame_events: Sequence[Dict[str, Any]], default_fps: float = 30.0) -> float:
    timestamps: List[float] = []
    for event in frame_events:
        value = event.get("receivedAtUnix", event.get("timestamp"))
        if value is not None:
            try:
                timestamps.append(float(value))
            except (TypeError, ValueError):
                pass
    if len(timestamps) < 2:
        return default_fps
    deltas = np.diff(np.asarray(timestamps, dtype=float))
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if len(deltas) == 0:
        return default_fps
    return float(1.0 / np.median(deltas))


def active_replay_path_from_server(
    server_url: str,
    timeout: float = 2.0,
    insecure: bool = True,
) -> Path:
    url = server_url.rstrip("/") + "/debug/skeleton"
    context = ssl._create_unverified_context() if insecure and url.startswith("https://") else None
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
            debug = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not query IOBT skeleton server at {url}: {exc}") from exc

    replay = debug.get("replay") or {}
    sessions = replay.get("activeSessions") or []
    if not sessions:
        raise RuntimeError(f"IOBT skeleton server at {url} has no active replay session")

    session = sorted(sessions, key=lambda item: item.get("startedAt", ""))[-1]
    path = session.get("path")
    if not path:
        raise RuntimeError("Active IOBT replay session does not report a path")
    return Path(path).expanduser()


class IOBTCanonicalProcessor:
    def __init__(
        self,
        required_joint_names: Sequence[str] = G1_23DOF_REQUIRED_JOINTS,
        add_hand_roll_targets: bool = True,
    ) -> None:
        self.required_joint_names = tuple(required_joint_names)
        self.add_hand_roll_targets = add_hand_roll_targets
        self.metadata: Optional[Dict[str, Any]] = None
        self.joint_defs: List[Dict[str, Any]] = []
        self.actual_human_height = DEFAULT_CANONICAL_HUMAN_HEIGHT

    def process_event(self, event: Dict[str, Any]) -> Optional[IOBTSkeletonFrame]:
        event_type = event.get("type")
        if event_type == "metadata":
            self.configure(event.get("metadata") or event)
            return None
        if event_type == "frame":
            return self.process_frame_event(event)
        return None

    def configure(self, metadata: Dict[str, Any]) -> None:
        validate_iobt_metadata(metadata, self.required_joint_names)
        self.metadata = copy.deepcopy(metadata)
        self.joint_defs = list(metadata["joints"])
        height_source = metadata.get("skeletonHeightSource")
        if height_source in CANONICAL_HEIGHT_SOURCE_OVERRIDES:
            self.actual_human_height = CANONICAL_HEIGHT_SOURCE_OVERRIDES[height_source]
        else:
            self.actual_human_height = float(
                metadata.get("skeletonHeightMeters") or DEFAULT_CANONICAL_HUMAN_HEIGHT
            )

    def process_frame_event(self, event: Dict[str, Any]) -> IOBTSkeletonFrame:
        if self.metadata is None:
            raise ValueError("IOBT frame arrived before metadata")

        frame_joints = event.get("joints")
        if not isinstance(frame_joints, list):
            raise ValueError("IOBT frame event must contain a joints list")
        if len(frame_joints) != len(self.joint_defs):
            raise ValueError(
                f"IOBT frame has {len(frame_joints)} joints but metadata defines "
                f"{len(self.joint_defs)}"
            )

        unity_positions, unity_rotations = self._reconstruct_unity_world_pose(frame_joints)
        human_data: Dict[str, List[np.ndarray]] = {}
        for index, joint_def in enumerate(self.joint_defs):
            name = joint_def["name"]
            gmr_position = unity_position_to_gmr(unity_positions[index])
            gmr_rotation = unity_rotation_to_gmr(unity_rotations[index].as_quat())
            human_data[name] = [gmr_position, _normalize_quat_wxyz(gmr_rotation)]

        add_shoulder_socket_targets(human_data)
        if self.add_hand_roll_targets:
            add_hand_roll_targets(human_data)

        return IOBTSkeletonFrame(
            human_data=human_data,
            sequence=event.get("sequence"),
            timestamp=_optional_float(event.get("timestamp")),
            received_at_unix=_optional_float(event.get("receivedAtUnix")),
        )

    def _reconstruct_unity_world_pose(
        self,
        frame_joints: Sequence[Dict[str, Any]],
    ) -> Tuple[List[np.ndarray], List[R]]:
        world_positions: List[Optional[np.ndarray]] = [None] * len(self.joint_defs)
        world_rotations: List[Optional[R]] = [None] * len(self.joint_defs)

        for index, joint_def in enumerate(self.joint_defs):
            frame_joint = frame_joints[index]
            if not isinstance(frame_joint, dict):
                raise ValueError(f"IOBT frame joint {index} is not an object")

            parent_index = int(joint_def.get("parentIndex", -1))
            if parent_index < 0:
                world_positions[index] = _vector3(frame_joint.get("position", [0.0, 0.0, 0.0]))
                world_rotations[index] = _rotation_from_xyzw(
                    frame_joint.get("rotation", [0.0, 0.0, 0.0, 1.0])
                )
                continue

            parent_position = world_positions[parent_index]
            parent_rotation = world_rotations[parent_index]
            if parent_position is None or parent_rotation is None:
                raise ValueError(f"IOBT frame joint {index} was processed before its parent")

            bind_local_position = _vector3(joint_def.get("bindLocalPosition", [0.0, 0.0, 0.0]))
            bind_local_rotation = _quat_xyzw(
                joint_def.get("bindLocalRotation", [0.0, 0.0, 0.0, 1.0])
            )
            local_rotation_xyzw = _quat_xyzw(frame_joint.get("rotation", bind_local_rotation))
            local_rotation = _rotation_from_xyzw(local_rotation_xyzw)

            world_positions[index] = parent_position + parent_rotation.apply(bind_local_position)
            world_rotations[index] = parent_rotation * local_rotation

        positions: List[np.ndarray] = []
        rotations: List[R] = []
        for index, (position, rotation) in enumerate(zip(world_positions, world_rotations)):
            if position is None or rotation is None:
                raise ValueError(f"IOBT frame joint {index} did not produce a world pose")
            positions.append(np.asarray(position, dtype=float))
            rotations.append(rotation)
        return positions, rotations


class IOBTSkeletonSource:
    def __init__(
        self,
        source: str,
        input_path: Optional[str] = None,
        server_url: str = "http://127.0.0.1:8765",
        replay_file: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        start_at_end: bool = False,
        poll_interval: float = 0.02,
        stop_on_replay_end: bool = False,
        add_hand_roll_targets: bool = True,
        insecure: bool = True,
    ) -> None:
        if source not in ("replay", "live"):
            raise ValueError("IOBTSkeletonSource source must be 'replay' or 'live'")
        self.source = source
        self.input_path = Path(input_path).expanduser() if input_path else None
        self.server_url = server_url
        self.replay_file = Path(replay_file).expanduser() if replay_file else None
        self.start = start
        self.end = end
        self.start_at_end = start_at_end
        self.poll_interval = poll_interval
        self.stop_on_replay_end = stop_on_replay_end
        self.insecure = insecure
        self.processor = IOBTCanonicalProcessor(add_hand_roll_targets=add_hand_roll_targets)
        self.frames: List[IOBTSkeletonFrame] = []
        self.fps = 30.0
        self._iterator: Optional[Iterator[IOBTSkeletonFrame]] = None

        if source == "replay":
            if self.input_path is None:
                raise ValueError("Replay IOBTSkeletonSource requires input_path")
            self.frames = self._load_replay(self.input_path, start, end)
        else:
            if self.replay_file is None and self.input_path is not None:
                self.replay_file = self.input_path
            if self.replay_file is None:
                self.replay_file = active_replay_path_from_server(server_url, insecure=insecure)
            if self.replay_file.exists():
                self._prime_metadata(self.replay_file)

    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        return self.processor.metadata

    @property
    def actual_human_height(self) -> float:
        return self.processor.actual_human_height

    def iter_frames(self) -> Iterator[IOBTSkeletonFrame]:
        if self.source == "replay":
            yield from self.frames
            return
        if self.replay_file is None:
            raise RuntimeError("Live IOBTSkeletonSource does not have a replay file to follow")
        yield from self._follow_live_file(self.replay_file)

    def get_processed_body_data(self, use_hands: bool = False) -> Optional[Dict[str, List[np.ndarray]]]:
        del use_hands
        if self._iterator is None:
            self._iterator = self.iter_frames()
        try:
            return copy_human_data(next(self._iterator).human_data)
        except StopIteration:
            return None

    def get_current_frame(self) -> Optional[Dict[str, List[np.ndarray]]]:
        return self.get_processed_body_data()

    def _load_replay(
        self,
        path: Path,
        start: Optional[int],
        end: Optional[int],
    ) -> List[IOBTSkeletonFrame]:
        frame_events: List[Dict[str, Any]] = []
        frames: List[IOBTSkeletonFrame] = []
        frame_index = 0
        for event in _read_jsonl_events(path):
            if event.get("type") == "frame":
                frame_events.append(event)
                if end is not None and frame_index >= end:
                    break
                should_keep = start is None or frame_index >= start
                frame = self.processor.process_event(event)
                if frame is not None and should_keep:
                    frames.append(frame)
                frame_index += 1
                continue

            self.processor.process_event(event)

        self.fps = infer_iobt_fps(frame_events)
        return frames

    def _follow_live_file(self, path: Path) -> Iterator[IOBTSkeletonFrame]:
        while not path.exists():
            time.sleep(self.poll_interval)

        with path.open("r", encoding="utf-8") as handle:
            if self.start_at_end:
                for line in handle:
                    event = _loads_event(line, path, None)
                    if event.get("type") == "metadata":
                        self.processor.process_event(event)
                handle.seek(0, 2)

            while True:
                line = handle.readline()
                if not line:
                    time.sleep(self.poll_interval)
                    continue
                event = _loads_event(line, path, None)
                if event.get("type") == "replay_end" and self.stop_on_replay_end:
                    return
                frame = self.processor.process_event(event)
                if frame is not None:
                    yield frame

    def _prime_metadata(self, path: Path) -> None:
        for event in _read_jsonl_events(path):
            if event.get("type") == "metadata":
                self.processor.process_event(event)
                return


def add_hand_roll_targets(human_data: Dict[str, List[np.ndarray]]) -> None:
    for side in ("Left", "Right"):
        lower_name = f"{side}LowerArm"
        hand_name = f"{side}Hand"
        target_name = f"{side}HandRoll"
        if lower_name not in human_data or hand_name not in human_data:
            continue
        lower_position = np.asarray(human_data[lower_name][0], dtype=float)
        hand_position = np.asarray(human_data[hand_name][0], dtype=float)
        hand_rotation = R.from_quat(human_data[hand_name][1], scalar_first=True)
        roll_rotation = _hand_roll_rotation(side, lower_position, hand_position, hand_rotation)
        human_data[target_name] = [
            hand_position.copy(),
            roll_rotation.as_quat(scalar_first=True),
        ]


def add_shoulder_socket_targets(human_data: Dict[str, List[np.ndarray]]) -> None:
    for side in ("Left", "Right"):
        upper_name = f"{side}UpperArm"
        target_name = f"{side}ShoulderSocket"
        if upper_name not in human_data:
            continue
        human_data[target_name] = [
            np.asarray(human_data[upper_name][0], dtype=float).copy(),
            np.asarray(human_data[upper_name][1], dtype=float).copy(),
        ]


def _hand_roll_rotation(
    side: str,
    lower_position: np.ndarray,
    hand_position: np.ndarray,
    hand_rotation: R,
) -> R:
    x_axis = _normalize(hand_position - lower_position)
    if x_axis is None:
        x_axis = _normalize(hand_rotation.apply(HAND_FORWARD_LOCAL_AXES.get(side, np.array([1.0, 0.0, 0.0]))))
    if x_axis is None:
        x_axis = HAND_FORWARD_LOCAL_AXIS.copy()

    palm_axis = HAND_PALM_NORMAL_LOCAL_AXES.get(side, np.array([0.0, 0.0, 1.0]))
    palm_normal = hand_rotation.apply(palm_axis)
    z_axis = _normalize(palm_normal - np.dot(palm_normal, x_axis) * x_axis)
    if z_axis is None:
        fallback = hand_rotation.apply(np.array([0.0, 1.0, 0.0]))
        z_axis = _normalize(fallback - np.dot(fallback, x_axis) * x_axis)
    if z_axis is None:
        z_axis = _orthogonal_axis(x_axis)

    y_axis = _normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        y_axis = _orthogonal_axis(x_axis)
        z_axis = _normalize(np.cross(x_axis, y_axis))
    else:
        z_axis = _normalize(np.cross(x_axis, y_axis))

    matrix = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(matrix) < 0.0:
        y_axis = -y_axis
        matrix = np.column_stack([x_axis, y_axis, z_axis])
    roll_rotation = R.from_matrix(matrix)
    correction_degrees = HAND_ROLL_LOCAL_CORRECTION_DEGREES.get(side, 0.0)
    if correction_degrees:
        roll_rotation = roll_rotation * R.from_euler("x", correction_degrees, degrees=True)
    return roll_rotation


def _read_jsonl_events(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield _loads_event(line, path, line_number)


def _loads_event(line: str, path: Path, line_number: Optional[int]) -> Dict[str, Any]:
    try:
        event = json.loads(line)
    except json.JSONDecodeError as exc:
        location = f"{path}:{line_number}" if line_number is not None else str(path)
        raise ValueError(f"{location}: invalid JSONL event") from exc
    if not isinstance(event, dict):
        location = f"{path}:{line_number}" if line_number is not None else str(path)
        raise ValueError(f"{location}: replay event is not an object")
    return event


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vector3(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = [value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0)]
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected Vector3-compatible value, got shape {array.shape}")
    return array


def _quat_xyzw(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = [value.get("x", 0.0), value.get("y", 0.0), value.get("z", 0.0), value.get("w", 1.0)]
    array = np.asarray(value, dtype=float)
    if array.shape != (4,):
        raise ValueError(f"Expected Quaternion xyzw-compatible value, got shape {array.shape}")
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return array / norm


def _rotation_from_xyzw(value: Any) -> R:
    return R.from_quat(_quat_xyzw(value))


def _normalize_quat_wxyz(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return array / norm


def _normalize(value: Sequence[float], epsilon: float = 1e-8) -> Optional[np.ndarray]:
    array = np.asarray(value, dtype=float)
    norm = np.linalg.norm(array)
    if not np.isfinite(norm) or norm < epsilon:
        return None
    return array / norm


def _orthogonal_axis(axis: np.ndarray) -> np.ndarray:
    candidate = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, candidate))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    orthogonal = np.cross(candidate, axis)
    normalized = _normalize(orthogonal)
    if normalized is None:
        return np.array([0.0, 1.0, 0.0])
    return normalized
