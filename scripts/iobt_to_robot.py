import argparse
import pickle
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting, RobotMotionViewer
from general_motion_retargeting.iobt_utils import IOBT_CANONICAL_SRC_HUMAN, IOBTSkeletonSource


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


def build_source(args: argparse.Namespace) -> IOBTSkeletonSource:
    if args.source == "replay":
        return IOBTSkeletonSource(
            source="replay",
            input_path=args.input,
            start=args.start,
            end=args.end,
            add_hand_roll_targets=not args.disable_hand_roll_targets,
        )

    return IOBTSkeletonSource(
        source="live",
        input_path=args.input,
        replay_file=args.replay_file,
        server_url=args.server_url,
        start_at_end=args.start_at_end,
        poll_interval=args.poll_interval,
        add_hand_roll_targets=not args.disable_hand_roll_targets,
        insecure=not args.verify_tls,
    )


def iter_source_frames(source: IOBTSkeletonSource, max_frames: Optional[int]):
    count = 0
    for frame in source.iter_frames():
        yield frame
        count += 1
        if max_frames is not None and count >= max_frames:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Retarget IOBT canonical skeleton replay/live data to GMR.")
    parser.add_argument("--source", choices=("replay", "live"), required=True)
    parser.add_argument("--input", default=None, help="Replay JSONL path. For live, follow this file directly.")
    parser.add_argument("--replay-file", default=None, help="Live JSONL file to follow instead of querying server.")
    parser.add_argument("--server_url", default="http://127.0.0.1:8765")
    parser.add_argument("--robot", choices=("unitree_g1_23dof",), default="unitree_g1_23dof")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", default="videos/iobt_canonical_g1_23dof.mp4")
    parser.add_argument("--rate_limit", action="store_true", help="Run viewer at the skeleton frame rate.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--start-at-end", action="store_true", help="For live file tailing, skip old frames.")
    parser.add_argument("--poll-interval", type=float, default=0.02)
    parser.add_argument("--verify-tls", action="store_true", help="Verify HTTPS certificate when querying server.")
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--damping", type=float, default=5e-1)
    parser.add_argument("--ik-mode", choices=("adaptive", "single-pass"), default="adaptive")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help="Print GMR model and task details.")
    parser.add_argument(
        "--disable-hand-roll-targets",
        action="store_true",
        help="Do not generate LeftHandRoll/RightHandRoll synthetic wrist-roll targets.",
    )
    args = parser.parse_args()

    if args.source == "replay" and not args.input:
        parser.error("--source replay requires --input")

    source = build_source(args)
    fps = int(round(args.fps if args.fps is not None else source.fps))
    fps = max(1, fps)

    retargeter = GeneralMotionRetargeting(
        src_human=IOBT_CANONICAL_SRC_HUMAN,
        tgt_robot=args.robot,
        actual_human_height=source.actual_human_height,
        solver=args.solver,
        damping=args.damping,
        ik_mode=args.ik_mode,
        max_iter=args.max_iter,
        verbose=args.verbose,
    )

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
    progress = None
    if args.source == "replay":
        total = len(source.frames)
        if args.max_frames is not None:
            total = min(total, args.max_frames)
        progress = tqdm(total=total, desc="Retargeting IOBT canonical replay")

    try:
        for frame in iter_source_frames(source, args.max_frames):
            qpos = retargeter.retarget(frame.human_data)
            qpos_list.append(qpos)
            if viewer is not None:
                viewer.step(qpos[:3], qpos[3:7], qpos[7:], rate_limit=args.rate_limit)
            elif args.source == "live":
                time.sleep(0.0)
            if progress is not None:
                progress.update(1)
    except KeyboardInterrupt:
        pass
    finally:
        if progress is not None:
            progress.close()
        if viewer is not None:
            viewer.close()

    if args.save_path and qpos_list:
        save_robot_motion(args.save_path, fps, qpos_list)
        print(f"Saved {len(qpos_list)} frames to {args.save_path}")


if __name__ == "__main__":
    main()
