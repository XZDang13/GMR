import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import mujoco as mj
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.g1_grounding import stabilize_g1_support_feet
from general_motion_retargeting.utils.iobt_g1_postprocess import G1IobtLowerBodyPostprocessor
from general_motion_retargeting.utils.iobt_replay import (
    G1_23DOF_REQUIRED_JOINTS,
    build_gmr_human_frames,
    infer_iobt_src_human,
    metadata_height,
    validate_iobt_metadata,
)
from scripts.iobt_replay_to_robot import (
    apply_robot_root_horizontal_source,
    compute_ground_offset,
    copy_human_data,
)


IK_MODES = ("adaptive", "single-pass")
ROBOT_ROOT_HORIZONTAL_SOURCES = ("human-root", "ik")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live retarget IOBT WebRTC receiver replay stream to a GMR robot."
    )
    parser.add_argument(
        "--server-url",
        default="https://127.0.0.1:8080",
        help="IOBT WebRTC receiver base URL.",
    )
    parser.add_argument(
        "--replay-file",
        default=None,
        help="Optional JSONL replay file to tail directly. If omitted, read active session path from /debug/skeleton.",
    )
    parser.add_argument(
        "--start-at-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For live sessions, read metadata then skip old frame backlog and tail new frames.",
    )
    parser.add_argument(
        "--robot",
        choices=["unitree_g1_23dof"],
        default="unitree_g1_23dof",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Viewer/update target FPS.")
    parser.add_argument("--no-viewer", action="store_true", help="Run live retargeting without opening the viewer.")
    parser.add_argument("--show-human", action="store_true", help="Draw the live human target frames in the MuJoCo viewer.")
    parser.add_argument("--rate_limit", action="store_true", help="Rate-limit viewer steps to --fps.")
    parser.add_argument(
        "--robot-root-horizontal-source",
        choices=ROBOT_ROOT_HORIZONTAL_SOURCES,
        default="human-root",
        help="human-root preserves the live IOBT root x/y instead of allowing free-base IK drift.",
    )
    parser.add_argument(
        "--no_robot_ground_normalization",
        action="store_true",
        help="Do not shift G1 root height to place the lowest support foot on the ground.",
    )
    parser.add_argument(
        "--ground-calibration-frames",
        type=int,
        default=50,
        help="Number of initial live frames used to estimate human ground offset before opening the viewer.",
    )
    parser.add_argument("--ik-mode", choices=IK_MODES, default="adaptive")
    parser.add_argument("--ik-max-iter", type=int, default=3)
    parser.add_argument("--ik-min-improvement", type=float, default=0.001)
    parser.add_argument("--task-weight-epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--knee-flexion-gain",
        type=float,
        default=1.10,
        help="IOBT/G1 postprocess gain for conservative knee flexion tracking.",
    )
    parser.add_argument("--status-interval", type=float, default=2.0)
    parser.add_argument("--insecure", action="store_true", default=True, help="Accept the receiver self-signed HTTPS cert.")
    parser.add_argument("--verbose-retargeter", action="store_true")
    return parser.parse_args()


def request_json(url, insecure=True, timeout=1.0):
    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def active_replay_path_from_server(server_url, insecure=True):
    debug = request_json(server_url.rstrip("/") + "/debug/skeleton", insecure=insecure)
    replay = debug.get("replay") or {}
    active_sessions = replay.get("activeSessions") or []
    if not active_sessions:
        return None, debug

    latest = max(active_sessions, key=lambda item: item.get("startedAt") or "")
    path = latest.get("path")
    return Path(path).expanduser() if path else None, debug


def wait_for_replay_path(args):
    if args.replay_file:
        return Path(args.replay_file).expanduser()

    next_log = 0.0
    while True:
        try:
            path, debug = active_replay_path_from_server(args.server_url, insecure=args.insecure)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            path = None
            debug = {"status": f"receiver unavailable: {exc}"}

        if path is not None:
            print(f"Using live replay file: {path}")
            return path

        now = time.monotonic()
        if now >= next_log:
            print(f"Waiting for active IOBT WebRTC replay session ({debug.get('status', '-')})")
            next_log = now + max(0.5, args.status_interval)
        time.sleep(0.1)


def follow_jsonl(path, start_at_end=True):
    metadata = None
    with path.open("r", encoding="utf-8") as handle:
        if start_at_end:
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "metadata":
                    metadata = event.get("metadata")

            if metadata is not None:
                yield "metadata", metadata
                handle.seek(0, 2)
            else:
                handle.seek(position if "position" in locals() else 0)

        while True:
            line = handle.readline()
            if not line:
                time.sleep(0.005)
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            if event_type == "metadata":
                metadata = event.get("metadata")
                yield event_type, metadata
            elif event_type == "frame":
                yield event_type, event
            elif event_type == "replay_end":
                yield event_type, event


def wait_for_existing_file(path):
    while not path.exists():
        print(f"Waiting for replay file to appear: {path}")
        time.sleep(0.25)


def frame_event_to_human_data(metadata, frame_event):
    return build_gmr_human_frames(metadata, [frame_event])[0]


def main():
    args = parse_args()
    motion_fps = max(1, int(round(args.fps)))

    print("Live IOBT -> GMR retarget")
    print("Root position mode: live (no freeze)")
    print(f"Robot root horizontal source: {args.robot_root_horizontal_source}")
    print(
        "IK: "
        f"mode={args.ik_mode}, max_iter={args.ik_max_iter}, "
        f"min_improvement={args.ik_min_improvement}, task_weight_epsilon={args.task_weight_epsilon}"
    )
    print(f"IOBT/G1 knee flexion gain: {args.knee_flexion_gain:.3f}")

    replay_path = wait_for_replay_path(args)
    wait_for_existing_file(replay_path)

    metadata = None
    retargeter = None
    lower_body_postprocessor = None
    viewer = None
    calibration_frames = []
    calibrated = False
    processed = 0
    start_time = time.monotonic()
    interval_processed = 0
    interval_start_time = start_time
    next_status = start_time + max(0.5, args.status_interval)

    try:
        for event_type, event in follow_jsonl(replay_path, start_at_end=args.start_at_end):
            if event_type == "metadata":
                metadata = event
                validate_iobt_metadata(metadata, G1_23DOF_REQUIRED_JOINTS)
                actual_human_height = metadata_height(metadata)
                src_human = infer_iobt_src_human(metadata)
                print(
                    f"Metadata ready: joints={len(metadata.get('joints') or [])}, "
                    f"height={actual_human_height}, encoding={metadata.get('poseEncoding')}, "
                    f"sourceConfig={src_human}"
                )
                retargeter = GMR(
                    src_human=src_human,
                    tgt_robot=args.robot,
                    actual_human_height=actual_human_height,
                    verbose=args.verbose_retargeter,
                    ik_mode=args.ik_mode,
                    max_iter=args.ik_max_iter,
                    min_improvement=args.ik_min_improvement,
                    task_weight_epsilon=args.task_weight_epsilon,
                )
                lower_body_postprocessor = G1IobtLowerBodyPostprocessor(
                    retargeter.model,
                    knee_flexion_gain=args.knee_flexion_gain,
                )
                continue

            if event_type == "replay_end":
                print("Replay session ended; live retarget stopped.")
                break

            if metadata is None or retargeter is None or lower_body_postprocessor is None:
                continue

            human_data = frame_event_to_human_data(metadata, event)
            if not calibrated and args.ground_calibration_frames > 0:
                calibration_frames.append(copy_human_data(human_data))
                if len(calibration_frames) < args.ground_calibration_frames:
                    continue

                ground_offset = compute_ground_offset(retargeter, calibration_frames)
                retargeter.set_ground_offset(ground_offset)
                calibrated = True
                print(
                    f"Ground offset calibrated from {len(calibration_frames)} live frames: "
                    f"{ground_offset:.4f}"
                )
            elif not calibrated:
                calibrated = True

            if viewer is None and not args.no_viewer:
                viewer = RobotMotionViewer(
                    robot_type=args.robot,
                    motion_fps=motion_fps,
                    transparent_robot=1,
                    camera_follow=False,
                )

            qpos = retargeter.retarget(copy_human_data(human_data))
            if not np.all(np.isfinite(qpos)):
                raise ValueError("Live retargeting produced non-finite qpos")

            qpos = lower_body_postprocessor.apply(qpos, human_data)

            if not args.no_robot_ground_normalization:
                qpos, lowest_support_height, _adjusted_feet = stabilize_g1_support_feet(
                    retargeter.model,
                    retargeter.configuration.data,
                    qpos,
                )
                if not np.isfinite(lowest_support_height):
                    raise ValueError("Robot foot grounding produced a non-finite support height")

            qpos = apply_robot_root_horizontal_source(
                qpos,
                human_data,
                args.robot_root_horizontal_source,
            )
            retargeter.configuration.data.qpos[:] = qpos
            mj.mj_forward(retargeter.model, retargeter.configuration.data)

            if viewer is not None:
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retargeter.scaled_human_data if args.show_human else None,
                    rate_limit=args.rate_limit,
                    follow_camera=False,
                )

            processed += 1
            interval_processed += 1
            now = time.monotonic()
            if now >= next_status:
                total_elapsed = max(0.0001, now - start_time)
                interval_elapsed = max(0.0001, now - interval_start_time)
                print(
                    f"Live retarget FPS: {interval_processed / interval_elapsed:.2f} "
                    f"(avg {processed / total_elapsed:.2f}); source sequence={event.get('sequence')}"
                )
                interval_processed = 0
                interval_start_time = now
                next_status = now + max(0.5, args.status_interval)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
