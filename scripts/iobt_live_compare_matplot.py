import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from rich import print

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.g1_grounding import stabilize_g1_support_feet
from general_motion_retargeting.utils.iobt_g1_postprocess import G1IobtLowerBodyPostprocessor
from general_motion_retargeting.utils.iobt_replay import (
    G1_23DOF_REQUIRED_JOINTS,
    infer_iobt_src_human,
    metadata_height,
    validate_iobt_metadata,
)
from scripts.iobt_live_retarget import (
    frame_event_to_human_data,
    follow_jsonl,
    wait_for_existing_file,
    wait_for_replay_path,
)
from scripts.iobt_replay_to_robot import (
    apply_robot_root_horizontal_source,
    compute_ground_offset,
    copy_human_data,
)


IK_MODES = ("adaptive", "single-pass")
ROBOT_ROOT_HORIZONTAL_SOURCES = ("human-root", "ik")

HUMAN_EDGES = (
    ("Hips", "Chest"),
    ("Chest", "LeftShoulder"),
    ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftLowerArm"),
    ("LeftLowerArm", "LeftHand"),
    ("Chest", "RightShoulder"),
    ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightLowerArm"),
    ("RightLowerArm", "RightHand"),
    ("Hips", "LeftUpperLeg"),
    ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"),
    ("LeftFoot", "LeftToes"),
    ("Hips", "RightUpperLeg"),
    ("RightUpperLeg", "RightLowerLeg"),
    ("RightLowerLeg", "RightFoot"),
    ("RightFoot", "RightToes"),
)

ROBOT_EDGES = (
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_rubber_hand_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_rubber_hand_link"),
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("left_ankle_roll_link", "left_toe_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("right_ankle_roll_link", "right_toe_link"),
)

LIMB_PAIRS = {
    "left_thigh": (("LeftUpperLeg", "LeftLowerLeg"), ("left_hip_roll_link", "left_knee_link")),
    "left_shin": (("LeftLowerLeg", "LeftFoot"), ("left_knee_link", "left_ankle_roll_link")),
    "right_thigh": (("RightUpperLeg", "RightLowerLeg"), ("right_hip_roll_link", "right_knee_link")),
    "right_shin": (("RightLowerLeg", "RightFoot"), ("right_knee_link", "right_ankle_roll_link")),
}


def body_positions(model, data, names):
    return {
        name: data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)].copy()
        for name in names
    }


def limb_angle_degrees(a0, a1, b0, b1):
    va = a1 - a0
    vb = b1 - b0
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm < 1e-9:
        return np.nan
    cosine = np.clip(np.dot(va, vb) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def sagittal_lean_degrees(lower_position, upper_position):
    delta = np.asarray(upper_position, dtype=float) - np.asarray(lower_position, dtype=float)
    return float(np.degrees(np.arctan2(delta[0], delta[2])))


def parse_args():
    parser = argparse.ArgumentParser(description="Live Matplotlib comparison for IOBT skeleton vs G1 retarget.")
    parser.add_argument("--server-url", default="https://127.0.0.1:8080")
    parser.add_argument("--replay-file", default=None)
    parser.add_argument("--start-at-end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot", choices=["unitree_g1_23dof"], default="unitree_g1_23dof")
    parser.add_argument("--plot-fps", type=float, default=12.0)
    parser.add_argument("--status-interval", type=float, default=2.0)
    parser.add_argument("--ground-calibration-frames", type=int, default=50)
    parser.add_argument("--robot-root-horizontal-source", choices=ROBOT_ROOT_HORIZONTAL_SOURCES, default="human-root")
    parser.add_argument("--no_robot_ground_normalization", action="store_true")
    parser.add_argument("--ik-mode", choices=IK_MODES, default="adaptive")
    parser.add_argument("--ik-max-iter", type=int, default=3)
    parser.add_argument("--ik-min-improvement", type=float, default=0.001)
    parser.add_argument("--task-weight-epsilon", type=float, default=1e-5)
    parser.add_argument("--knee-flexion-gain", type=float, default=1.10)
    parser.add_argument("--insecure", action="store_true", default=True)
    parser.add_argument("--verbose-retargeter", action="store_true")
    return parser.parse_args()


def draw_edges(ax, positions, edges, color, label, marker="o"):
    first = True
    for a, b in edges:
        if a not in positions or b not in positions:
            continue
        pa = positions[a]
        pb = positions[b]
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            color=color,
            linewidth=2.2,
            marker=marker,
            markersize=3.2,
            label=label if first else None,
        )
        first = False


def lower_body_angle_summary(human_positions, robot_positions):
    parts = []
    for limb_name in ("left_thigh", "left_shin", "right_thigh", "right_shin"):
        (ha, hb), (ra, rb) = LIMB_PAIRS[limb_name]
        if ha not in human_positions or hb not in human_positions or ra not in robot_positions or rb not in robot_positions:
            continue
        angle = limb_angle_degrees(human_positions[ha], human_positions[hb], robot_positions[ra], robot_positions[rb])
        if np.isfinite(angle):
            parts.append(f"{limb_name}={angle:.0f}deg")
    return "  ".join(parts)


class LiveMatplotCompare:
    def __init__(self, plot_fps):
        self.plot_period = 1.0 / max(1.0, float(plot_fps))
        self.next_plot_time = 0.0
        plt.ion()
        self.fig = plt.figure(figsize=(9.6, 7.2))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.canvas.manager.set_window_title("IOBT live skeleton vs G1 retarget")
        self.fig.show()

    def maybe_update(self, human_positions, robot_positions, sequence, retarget_fps):
        now = time.monotonic()
        if now < self.next_plot_time:
            return
        self.next_plot_time = now + self.plot_period

        root = robot_positions.get("pelvis")
        if root is None:
            root = human_positions.get("Hips", np.zeros(3))

        human_torso_lean = sagittal_lean_degrees(human_positions["Hips"], human_positions["Chest"])
        robot_torso_lean = sagittal_lean_degrees(robot_positions["pelvis"], robot_positions["torso_link"])
        lower_summary = lower_body_angle_summary(human_positions, robot_positions)

        self.ax.clear()
        self.ax.view_init(elev=14, azim=-62)
        self.ax.set_xlim(root[0] - 1.1, root[0] + 1.1)
        self.ax.set_ylim(root[1] - 1.1, root[1] + 1.1)
        self.ax.set_zlim(-0.15, 1.9)
        self.ax.set_xlabel("GMR X forward")
        self.ax.set_ylabel("GMR Y left")
        self.ax.set_zlabel("GMR Z up")
        self.ax.set_title(
            f"seq={sequence}  retarget={retarget_fps:.1f} fps  "
            f"lean human={human_torso_lean:+.1f} robot={robot_torso_lean:+.1f}\n{lower_summary}"
        )
        draw_edges(self.ax, human_positions, HUMAN_EDGES, "#00a6ff", "IOBT skeleton")
        draw_edges(self.ax, robot_positions, ROBOT_EDGES, "#ff4b4b", "G1 retarget", marker="^")
        self.ax.legend(loc="upper right")
        self.ax.grid(True, alpha=0.25)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main():
    args = parse_args()
    print("Live IOBT -> GMR Matplotlib comparison")
    print("Root position mode: live (no freeze)")
    print(f"IK: mode={args.ik_mode}, max_iter={args.ik_max_iter}, epsilon={args.task_weight_epsilon}")
    print(f"Plot FPS target: {args.plot_fps}")

    replay_path = wait_for_replay_path(args)
    wait_for_existing_file(replay_path)

    metadata = None
    retargeter = None
    lower_body_postprocessor = None
    calibration_frames = []
    calibrated = False
    processed = 0
    interval_processed = 0
    start_time = time.monotonic()
    interval_start = start_time
    next_status = start_time + max(0.5, args.status_interval)
    plotter = LiveMatplotCompare(args.plot_fps)
    robot_body_names = sorted({name for edge in ROBOT_EDGES for name in edge})

    for event_type, event in follow_jsonl(replay_path, start_at_end=args.start_at_end):
        if not plt.fignum_exists(plotter.fig.number):
            print("Matplotlib window closed; live comparison stopped.")
            break

        if event_type == "metadata":
            metadata = event
            validate_iobt_metadata(metadata, G1_23DOF_REQUIRED_JOINTS)
            actual_human_height = metadata_height(metadata)
            src_human = infer_iobt_src_human(metadata)
            print(
                f"Metadata ready: source={metadata.get('source')}, "
                f"encoding={metadata.get('poseEncoding')}, height={actual_human_height}, "
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
            print("Replay session ended; live comparison stopped.")
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
            print(f"Ground offset calibrated from {len(calibration_frames)} frames: {ground_offset:.4f}")
        elif not calibrated:
            calibrated = True

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

        qpos = apply_robot_root_horizontal_source(qpos, human_data, args.robot_root_horizontal_source)
        retargeter.configuration.data.qpos[:] = qpos
        mj.mj_forward(retargeter.model, retargeter.configuration.data)

        human_positions = {
            name: np.asarray(pos, dtype=float).copy()
            for name, (pos, _quat) in retargeter.scaled_human_data.items()
        }
        robot_positions = body_positions(
            retargeter.model,
            retargeter.configuration.data,
            robot_body_names,
        )

        processed += 1
        interval_processed += 1
        now = time.monotonic()
        interval_elapsed = max(0.0001, now - interval_start)
        retarget_fps = interval_processed / interval_elapsed
        plotter.maybe_update(human_positions, robot_positions, event.get("sequence"), retarget_fps)

        if now >= next_status:
            total_elapsed = max(0.0001, now - start_time)
            print(
                f"Live compare FPS: {retarget_fps:.2f} "
                f"(avg {processed / total_elapsed:.2f}); source sequence={event.get('sequence')}"
            )
            interval_processed = 0
            interval_start = now
            next_status = now + max(0.5, args.status_interval)


if __name__ == "__main__":
    main()
