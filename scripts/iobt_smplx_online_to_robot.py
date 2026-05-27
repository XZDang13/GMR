import argparse
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer
from iobt_smplx_motion_to_robot import (
    UNITY_TO_GMR,
    disable_orientation_costs,
    estimate_human_height,
    first_joint_rotations,
    frame_pelvis_position,
    normalize_quat_wxyz,
    repair_legacy_smplx_rotation,
    save_robot_motion,
    unity_rotation_to_gmr,
    uses_smplx_rotation_frame,
    vector3,
)


DEFAULT_SERVER_URL = "https://127.0.0.1:8765"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Online retarget IOBT server SMPL-X body motion through GMR. "
            "Use --mode render for the viewer or --mode headless for compute/save only."
        )
    )
    parser.add_argument("--source", choices=("live", "replay"), default="live")
    parser.add_argument("--mode", choices=("headless", "render"), default="render")
    parser.add_argument("--motion-file", default=None, help="SMPL-X motion JSONL to read or follow.")
    parser.add_argument("--server_url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--verify-tls", action="store_true", help="Verify server TLS certificate.")
    parser.add_argument(
        "--session-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for an active server SMPL-X motion session; 0 waits forever.",
    )
    parser.add_argument("--poll-interval", type=float, default=0.02)
    parser.add_argument(
        "--start-at-end",
        action="store_true",
        help="For live files, skip existing records and retarget only newly appended frames.",
    )
    parser.add_argument(
        "--stop-on-motion-end",
        action="store_true",
        help="Stop live mode when a smplx_motion_end record is appended.",
    )
    parser.add_argument("--robot", choices=("unitree_g1_23dof",), default="unitree_g1_23dof")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/iobt_smplx_online_g1_23dof.mp4")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--height",
        type=float,
        default=1.8,
        help="Human height passed to GMR. Defaults to the SMPL-X IK config assumption.",
    )
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--damping", type=float, default=5e-1)
    parser.add_argument("--ik-mode", choices=("adaptive", "single-pass"), default="adaptive")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--status-interval", type=float, default=2.0)
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Disable orientation IK costs; useful when source joint axes are not trusted.",
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
            "the first streamed frame is not a bind pose."
        ),
    )
    parser.add_argument(
        "--root-origin",
        action="store_true",
        help="Subtract the first streamed pelvis XY position after converting to GMR coordinates.",
    )
    args = parser.parse_args()

    source = IOBTSmplxMotionSource(
        source=args.source,
        motion_file=Path(args.motion_file).expanduser() if args.motion_file else None,
        server_url=args.server_url,
        verify_tls=args.verify_tls,
        session_timeout=args.session_timeout,
        poll_interval=args.poll_interval,
        start_at_end=args.start_at_end,
        stop_on_motion_end=args.stop_on_motion_end,
        start=args.start,
        end=args.end,
        identity_rotations=args.identity_rotations,
        repair_legacy_rotation_frame=not args.disable_legacy_frame_correction,
        calibrate_initial_rotations=args.calibrate_initial_rotations,
        root_origin=args.root_origin,
    )

    frames_iter = iter(source)
    first_frame = next_or_raise(frames_iter, "No smplx_motion_frame records were available.")
    fps = max(1, int(round(args.fps)))
    human_height = args.height if args.height is not None else estimate_human_height([first_frame])

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
    if args.mode == "render":
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=fps,
            camera_follow=False,
            record_video=args.record_video,
            video_path=args.video_path,
        )

    qpos_list: List[np.ndarray] = []
    status = StatusPrinter(interval=args.status_interval)
    frame_count = 0
    started_at = time.monotonic()
    try:
        for frame in chain_first(first_frame, frames_iter):
            qpos = retargeter.retarget(frame["human_data"])
            qpos_list.append(qpos.copy())
            frame_count += 1

            if viewer is not None:
                viewer.step(
                    qpos[:3],
                    qpos[3:7],
                    qpos[7:],
                    human_motion_data=retargeter.scaled_human_data,
                    rate_limit=args.rate_limit,
                    follow_camera=False,
                )
            else:
                status.maybe_print(frame_count, frame, started_at)

            if args.max_frames is not None and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted; saving collected motion before exit if --save_path was provided.")
    finally:
        if viewer is not None:
            viewer.close()

    if args.save_path and qpos_list:
        save_robot_motion(args.save_path, fps, qpos_list)
        print(f"Saved {len(qpos_list)} frames to {args.save_path}")
    print(f"Retargeted {frame_count} SMPL-X frames from {source.motion_file}")


class IOBTSmplxMotionSource:
    def __init__(
        self,
        source: str,
        motion_file: Optional[Path],
        server_url: str,
        verify_tls: bool,
        session_timeout: float,
        poll_interval: float,
        start_at_end: bool,
        stop_on_motion_end: bool,
        start: Optional[int],
        end: Optional[int],
        identity_rotations: bool,
        repair_legacy_rotation_frame: bool,
        calibrate_initial_rotations: bool,
        root_origin: bool,
    ) -> None:
        self.source = source
        self._motion_file = motion_file.expanduser().resolve() if motion_file else None
        self.server_url = server_url
        self.verify_tls = verify_tls
        self.session_timeout = session_timeout
        self.poll_interval = poll_interval
        self.start_at_end = start_at_end
        self.stop_on_motion_end = stop_on_motion_end
        self.start = start
        self.end = end
        self.identity_rotations = identity_rotations
        self.repair_legacy_rotation_frame = repair_legacy_rotation_frame
        self.calibrate_initial_rotations = calibrate_initial_rotations
        self.root_origin = root_origin
        self._root_xy: Optional[np.ndarray] = None
        self._initial_rotations: Optional[Dict[str, R]] = None

    @property
    def motion_file(self) -> Path:
        if self._motion_file is None:
            self._motion_file = wait_for_active_smplx_motion_path(
                self.server_url,
                verify_tls=self.verify_tls,
                poll_interval=self.poll_interval,
                session_timeout=self.session_timeout,
            )
            print(f"Following active SMPL-X motion stream: {self._motion_file}")
        return self._motion_file

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raw_iter = self._follow_live_file() if self.source == "live" else self._read_replay_file()
        for raw_frame in raw_iter:
            frame = self._convert_raw_frame(raw_frame)
            if frame is not None:
                yield frame

    def _read_replay_file(self) -> Iterator[Dict[str, Any]]:
        frame_index = 0
        with self.motion_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                event = parse_jsonl_event(line, self.motion_file, line_number)
                if event is None:
                    continue
                if event.get("type") != "smplx_motion_frame":
                    continue
                if self.end is not None and frame_index >= self.end:
                    return
                if self.start is None or frame_index >= self.start:
                    yield event
                frame_index += 1

    def _follow_live_file(self) -> Iterator[Dict[str, Any]]:
        path = self.motion_file
        while not path.exists():
            time.sleep(self.poll_interval)

        frame_index = 0
        with path.open("r", encoding="utf-8") as handle:
            if self.start_at_end:
                handle.seek(0, 2)

            while True:
                line = handle.readline()
                if not line:
                    time.sleep(self.poll_interval)
                    continue

                event = parse_jsonl_event(line, path, None)
                if event is None:
                    continue

                event_type = event.get("type")
                if event_type == "smplx_motion_end" and self.stop_on_motion_end:
                    return
                if event_type != "smplx_motion_frame":
                    continue

                if self.end is not None and frame_index >= self.end:
                    return
                if self.start is None or frame_index >= self.start:
                    yield event
                frame_index += 1

    def _convert_raw_frame(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._initial_rotations is None:
            self._initial_rotations = first_joint_rotations(raw) if self.calibrate_initial_rotations else {}

        if self._root_xy is None:
            if self.root_origin:
                root_pos = frame_pelvis_position(raw)
                self._root_xy = (UNITY_TO_GMR @ root_pos)[:2]
            else:
                self._root_xy = np.zeros(2, dtype=float)

        human_data = {}
        for joint in raw.get("joints", []):
            name = joint.get("smplxName")
            if not name:
                continue

            position = UNITY_TO_GMR @ vector3(joint.get("worldPosition", [0.0, 0.0, 0.0]))
            position[:2] -= self._root_xy

            if self.identity_rotations:
                quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                rotation = unity_rotation_to_gmr(joint.get("worldRotationQuatXyzw", [0.0, 0.0, 0.0, 1.0]))
                if self.repair_legacy_rotation_frame and not uses_smplx_rotation_frame(raw, joint):
                    rotation = repair_legacy_smplx_rotation(name, rotation)
                initial = self._initial_rotations.get(name)
                if initial is not None:
                    rotation = rotation * initial.inv()
                quat_wxyz = rotation.as_quat(scalar_first=True)

            human_data[name] = [position, normalize_quat_wxyz(quat_wxyz)]

        if not human_data:
            return None

        return {
            "human_data": human_data,
            "receivedAtUnix": raw.get("receivedAtUnix"),
            "timestamp": raw.get("timestamp"),
            "sequence": raw.get("sequence"),
            "sessionId": raw.get("sessionId"),
        }


class StatusPrinter:
    def __init__(self, interval: float) -> None:
        self.interval = max(0.1, interval)
        self._last_print = 0.0

    def maybe_print(self, frame_count: int, frame: Dict[str, Any], started_at: float) -> None:
        now = time.monotonic()
        if now - self._last_print < self.interval:
            return
        self._last_print = now
        elapsed = max(1e-6, now - started_at)
        source_sequence = frame.get("sequence")
        suffix = f", source sequence {source_sequence}" if source_sequence is not None else ""
        print(f"Retargeted {frame_count} frames ({frame_count / elapsed:.1f} fps{suffix})", flush=True)


def wait_for_active_smplx_motion_path(
    server_url: str,
    verify_tls: bool,
    poll_interval: float,
    session_timeout: float,
) -> Path:
    deadline = None if session_timeout <= 0.0 else time.monotonic() + session_timeout
    last_error: Optional[Exception] = None
    while True:
        try:
            return active_smplx_motion_path_from_server(server_url, verify_tls=verify_tls)
        except Exception as exc:
            last_error = exc

        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for an active SMPL-X motion session. "
                "Start the IOBT server with --save-smplx-motion and connect the Quest, "
                "or pass --motion-file explicitly. "
                f"Last error: {last_error}"
            ) from last_error
        time.sleep(poll_interval)


def active_smplx_motion_path_from_server(server_url: str, verify_tls: bool) -> Path:
    url = server_url.rstrip("/") + "/debug/skeleton"
    context = None
    if url.startswith("https://") and not verify_tls:
        context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(url, timeout=2.0, context=context) as response:
            debug = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not query {url}: {exc}") from exc

    smplx_motion = debug.get("smplxMotion") or {}
    if not smplx_motion.get("enabled"):
        raise RuntimeError("Server SMPL-X motion recorder is disabled; restart it with --save-smplx-motion.")

    sessions = [session for session in smplx_motion.get("activeSessions", []) if session.get("path")]
    if not sessions:
        raise RuntimeError("Server has no active SMPL-X motion session yet.")

    latest = max(sessions, key=lambda session: str(session.get("startedAt") or ""))
    return Path(latest["path"]).expanduser().resolve()


def parse_jsonl_event(line: str, path: Path, line_number: Optional[int]) -> Optional[Dict[str, Any]]:
    if not line.strip():
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError as exc:
        location = f"{path}:{line_number}" if line_number is not None else str(path)
        raise ValueError(f"{location}: invalid JSON") from exc
    if not isinstance(event, dict):
        return None
    return event


def next_or_raise(iterator: Iterator[Dict[str, Any]], message: str) -> Dict[str, Any]:
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError(message) from exc


def chain_first(first: Dict[str, Any], rest: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    yield first
    yield from rest


if __name__ == "__main__":
    main()
