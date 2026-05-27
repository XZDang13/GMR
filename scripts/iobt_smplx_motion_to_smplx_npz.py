import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R


UNITY_TO_GMR = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)
GMR_TO_UNITY = UNITY_TO_GMR.T


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert IOBT SMPL-X body-motion JSONL to GMR SMPL-X npz."
    )
    parser.add_argument("--input", required=True, help="IOBT smplx_motion JSONL file.")
    parser.add_argument("--output", required=True, help="Output SMPL-X npz path.")
    parser.add_argument("--gender", default="neutral", choices=("neutral", "male", "female"))
    parser.add_argument("--fps", type=float, default=None, help="Override mocap frame rate.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--keep-unity-coordinates",
        action="store_true",
        help="Keep Unity +Y-up coordinates instead of converting to GMR/MuJoCo +Z-up.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    frames = list(_iter_motion_frames(input_path, args.max_frames))
    if not frames:
        raise ValueError(f"No smplx_motion_frame records found in {input_path}")

    trans = []
    root_orient = []
    pose_body = []
    timestamps = []
    received_at = []

    for frame in frames:
        trans.append(_vector3(frame["transl"]))
        root_orient.append(_vector3(frame["globalOrientAxisAngle"]))
        body_pose = np.asarray(frame["bodyPoseAxisAngle"], dtype=np.float32)
        if body_pose.shape != (21, 3):
            raise ValueError(
                f"Expected bodyPoseAxisAngle shape (21, 3), got {body_pose.shape}"
            )
        pose_body.append(body_pose.reshape(-1))

        if frame.get("timestamp") is not None:
            timestamps.append(float(frame["timestamp"]))
        if frame.get("receivedAtUnix") is not None:
            received_at.append(float(frame["receivedAtUnix"]))

    trans_array = np.asarray(trans, dtype=np.float32)
    root_orient_array = np.asarray(root_orient, dtype=np.float32)
    pose_body_array = np.asarray(pose_body, dtype=np.float32)

    if not args.keep_unity_coordinates:
        trans_array = np.asarray([UNITY_TO_GMR @ value for value in trans_array], dtype=np.float32)
        root_orient_array = _rotvecs_unity_to_gmr(root_orient_array)
        pose_body_array = _rotvecs_unity_to_gmr(pose_body_array.reshape(-1, 3)).reshape(
            pose_body_array.shape
        )

    fps = args.fps
    if fps is None:
        fps = _infer_fps(received_at) or _infer_fps(timestamps) or 30.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        trans=trans_array,
        root_orient=root_orient_array,
        pose_body=pose_body_array,
        betas=np.zeros(16, dtype=np.float32),
        gender=np.array(args.gender),
        mocap_frame_rate=np.array(float(fps), dtype=np.float32),
        source_format=np.array("iobt-smplx-body-motion-jsonl"),
        source_path=np.array(str(input_path)),
    )

    print(
        f"Saved {len(frames)} frames at {fps:.3f} fps to {output_path} "
        f"(trans={trans_array.shape}, root_orient={root_orient_array.shape}, "
        f"pose_body={pose_body_array.shape})"
    )


def _iter_motion_frames(path: Path, max_frames: Optional[int]) -> Iterable[Dict[str, Any]]:
    count = 0
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
            yield event
            count += 1
            if max_frames is not None and count >= max_frames:
                return


def _vector3(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"Expected vector shape (3,), got {array.shape}")
    return array


def _rotvecs_unity_to_gmr(rotvecs: np.ndarray) -> np.ndarray:
    matrices = R.from_rotvec(rotvecs).as_matrix()
    converted = UNITY_TO_GMR @ matrices @ GMR_TO_UNITY
    return R.from_matrix(converted).as_rotvec().astype(np.float32)


def _infer_fps(times: List[float]) -> Optional[float]:
    if len(times) < 2:
        return None
    values = np.asarray(times, dtype=float)
    deltas = np.diff(values)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if len(deltas) == 0:
        return None
    return float(1.0 / np.median(deltas))


if __name__ == "__main__":
    main()
