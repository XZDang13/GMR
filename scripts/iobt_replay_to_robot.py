import argparse
import os
import pathlib
import pickle
import time

import mujoco as mj
import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.iobt_replay import (
    G1_23DOF_REQUIRED_JOINTS,
    ROOT_POSITION_MODES,
    infer_iobt_src_human,
    load_iobt_replay_file,
)
from general_motion_retargeting.utils.g1_grounding import stabilize_g1_support_feet
from general_motion_retargeting.utils.iobt_g1_postprocess import G1IobtLowerBodyPostprocessor


ROBOT_ROOT_HORIZONTAL_SOURCES = ("human-root", "ik")
IK_MODES = ("adaptive", "single-pass")
SRC_HUMAN_CHOICES = ("auto", "iobt_mocopi", "iobt_ovr_fullbody")


def copy_human_data(human_data):
    return {
        body_name: [np.asarray(pos, dtype=float).copy(), np.asarray(quat, dtype=float).copy()]
        for body_name, (pos, quat) in human_data.items()
    }


def compute_ground_offset(retargeter: GMR, motion_data, reference_frame_count=None):
    reference_motion = motion_data
    if reference_frame_count is not None and reference_frame_count > 0:
        reference_motion = motion_data[:reference_frame_count]

    lowest_per_frame = []
    for human_data in reference_motion:
        frame = copy_human_data(human_data)
        frame = retargeter.to_numpy(frame)
        frame = retargeter.scale_human_data(
            frame,
            retargeter.human_root_name,
            retargeter.human_scale_table,
        )
        frame = retargeter.offset_human_data(
            frame,
            retargeter.pos_offsets1,
            retargeter.rot_offsets1,
            retargeter.root_frame_pos_offsets,
        )
        frame_lowest = np.inf
        for body_name, (pos, _quat) in frame.items():
            if "Foot" not in body_name and "Toes" not in body_name:
                continue
            frame_lowest = min(frame_lowest, float(pos[2]))
        if np.isfinite(frame_lowest):
            lowest_per_frame.append(frame_lowest)

    if not lowest_per_frame:
        return 0.0

    return float(np.median(lowest_per_frame))


def save_robot_motion(save_path, motion_fps, qpos_list):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    # GMR/MuJoCo qpos stores root quaternion as wxyz; saved robot motion uses xyzw.
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])

    motion_data = {
        "fps": motion_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(save_path, "wb") as handle:
        pickle.dump(motion_data, handle)


def apply_robot_root_horizontal_source(qpos, human_data, source):
    if source == "ik":
        return qpos
    if source != "human-root":
        raise ValueError(f"Unknown robot root horizontal source: {source}")
    if "Hips" not in human_data:
        raise ValueError("Cannot drive robot root from human root: Hips joint is missing")

    qpos = qpos.copy()
    qpos[:2] = np.asarray(human_data["Hips"][0], dtype=float)[:2]
    return qpos


def parse_args():
    parser = argparse.ArgumentParser(description="Retarget an IOBT mocopi canonical JSONL replay to a GMR robot.")
    parser.add_argument("--replay_file", required=True, type=str, help="IOBT skeleton replay JSONL file.")
    parser.add_argument(
        "--src-human",
        choices=SRC_HUMAN_CHOICES,
        default="auto",
        help="GMR source config. auto selects iobt_ovr_fullbody for OVR UnityWorld replays.",
    )
    parser.add_argument(
        "--robot",
        choices=["unitree_g1_23dof"],
        default="unitree_g1_23dof",
        help="Target robot. First IOBT config supports G1 23DOF only.",
    )
    parser.add_argument("--save_path", default=None, help="Path to save retargeted robot motion pickle.")
    parser.add_argument("--no-viewer", action="store_true", help="Run retargeting without opening the MuJoCo viewer.")
    parser.add_argument("--record_video", action="store_true", default=False, help="Record the MuJoCo viewer output.")
    parser.add_argument("--video_path", type=str, default="videos/iobt_mocopi_g1_23dof.mp4")
    parser.add_argument("--rate_limit", action="store_true", default=False, help="Render at replay frame rate.")
    parser.add_argument("--start", default=None, type=int, help="First replay frame index to process.")
    parser.add_argument("--end", default=None, type=int, help="End replay frame index, exclusive.")
    parser.add_argument("--fps", default=None, type=float, help="Override replay FPS.")
    parser.add_argument(
        "--root-position-mode",
        choices=ROOT_POSITION_MODES,
        default="freeze-horizontal",
        help=(
            "How to handle replay root translation before retargeting. "
            "freeze-horizontal removes per-frame horizontal root drift while preserving height."
        ),
    )
    parser.add_argument(
        "--no_robot_ground_normalization",
        action="store_true",
        help="Do not shift G1 root height to place the lowest foot support point on the ground.",
    )
    parser.add_argument(
        "--robot-root-horizontal-source",
        choices=ROBOT_ROOT_HORIZONTAL_SOURCES,
        default="human-root",
        help=(
            "Where saved/viewed robot root x/y comes from. human-root prevents free-base IK drift "
            "while preserving the replay root translation mode."
        ),
    )
    parser.add_argument(
        "--ik-mode",
        choices=IK_MODES,
        default="adaptive",
        help="IK solve mode. single-pass is the low-latency realtime path; adaptive preserves the older iterative solve.",
    )
    parser.add_argument(
        "--ik-max-iter",
        type=int,
        default=10,
        help="Maximum extra adaptive IK iterations after the first solve. Ignored by single-pass.",
    )
    parser.add_argument(
        "--ik-min-improvement",
        type=float,
        default=0.001,
        help="Adaptive IK stops when error improvement falls below this value.",
    )
    parser.add_argument(
        "--task-weight-epsilon",
        type=float,
        default=0.0,
        help=(
            "Skip creating IK tasks whose position and rotation weights are at or below this value. "
            "Offsets are still preserved, so values like 1e-5 can drop tiny bookkeeping tasks."
        ),
    )
    parser.add_argument(
        "--knee-flexion-gain",
        type=float,
        default=1.10,
        help=(
            "IOBT/G1 lower-body postprocess gain. Values above 1.0 compensate the conservative "
            "G1 knee bend produced by position/orientation IK."
        ),
    )
    parser.add_argument(
        "--verbose-retargeter",
        action="store_true",
        help="Print robot model DoF/body/actuator tables while constructing the GMR retargeter.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_viewer and args.record_video:
        raise SystemExit("--record_video requires the viewer; remove --no-viewer")

    replay = load_iobt_replay_file(
        args.replay_file,
        start=args.start,
        end=args.end,
        required_joint_names=G1_23DOF_REQUIRED_JOINTS,
        root_position_mode=args.root_position_mode,
    )
    if not replay.frames:
        raise SystemExit("Replay selection produced no frames")

    motion_fps = int(round(args.fps if args.fps is not None else replay.fps))
    motion_fps = max(1, motion_fps)
    actual_human_height = replay.actual_human_height

    print(f"Loaded IOBT replay: {args.replay_file}")
    print(f"Frames: {len(replay.frames)}")
    print(f"Replay FPS: {replay.fps:.3f}; using {motion_fps}")
    print(f"Canonical height: {actual_human_height}")
    src_human = infer_iobt_src_human(replay.metadata) if args.src_human == "auto" else args.src_human
    print(f"GMR source config: {src_human}")
    print(f"Root position mode: {args.root_position_mode}")
    print(f"Robot root horizontal source: {args.robot_root_horizontal_source}")
    print(
        "IK: "
        f"mode={args.ik_mode}, max_iter={args.ik_max_iter}, "
        f"min_improvement={args.ik_min_improvement}, task_weight_epsilon={args.task_weight_epsilon}"
    )
    print(f"IOBT/G1 knee flexion gain: {args.knee_flexion_gain:.3f}")

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
    ground_offset = compute_ground_offset(retargeter, replay.frames, reference_frame_count=motion_fps)
    retargeter.set_ground_offset(ground_offset)
    print(f"Ground offset: {ground_offset:.4f}")

    viewer = None
    if not args.no_viewer:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=motion_fps,
            transparent_robot=1,
            record_video=args.record_video,
            video_path=args.video_path,
            camera_follow=False,
        )

    qpos_list = []
    robot_ground_shifts = []
    stabilized_foot_count = 0
    pbar = tqdm(total=len(replay.frames), desc="Retargeting IOBT replay")
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    try:
        for human_data in replay.frames:
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual retarget FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

            qpos = retargeter.retarget(copy_human_data(human_data))
            if not np.all(np.isfinite(qpos)):
                raise ValueError("Retargeting produced non-finite qpos")

            qpos = lower_body_postprocessor.apply(qpos, human_data)

            if not args.no_robot_ground_normalization:
                qpos_before_grounding = qpos.copy()
                qpos, lowest_support_height, adjusted_feet = stabilize_g1_support_feet(
                    retargeter.model,
                    retargeter.configuration.data,
                    qpos,
                )
                robot_ground_shifts.append(float(qpos[2] - qpos_before_grounding[2]))
                stabilized_foot_count += adjusted_feet
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
                    human_motion_data=retargeter.scaled_human_data,
                    rate_limit=args.rate_limit,
                    follow_camera=False,
                )

            if args.save_path is not None:
                qpos_list.append(qpos.copy())

            pbar.update(1)
    finally:
        pbar.close()
        if viewer is not None:
            viewer.close()

    if args.save_path is not None:
        save_robot_motion(args.save_path, motion_fps, qpos_list)
        print(f"Saved to {args.save_path}")

    if robot_ground_shifts:
        print(f"Robot ground normalization median root-z shift: {np.median(robot_ground_shifts):.4f}")
        print(f"Support-foot stabilization adjustments: {stabilized_foot_count}")


if __name__ == "__main__":
    main()
