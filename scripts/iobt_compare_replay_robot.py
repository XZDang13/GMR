import argparse
import sys
from pathlib import Path

import imageio
import matplotlib
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.iobt_replay import (
    G1_23DOF_REQUIRED_JOINTS,
    infer_iobt_src_human,
    load_iobt_replay_file,
)
from general_motion_retargeting.utils.g1_grounding import stabilize_g1_support_feet
from general_motion_retargeting.utils.iobt_g1_postprocess import G1IobtLowerBodyPostprocessor
from scripts.iobt_replay_to_robot import compute_ground_offset, copy_human_data


matplotlib.use("Agg")


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
    "left_upper_arm": (("LeftUpperArm", "LeftLowerArm"), ("left_shoulder_yaw_link", "left_elbow_link")),
    "left_forearm": (("LeftLowerArm", "LeftHand"), ("left_elbow_link", "left_rubber_hand_link")),
    "right_upper_arm": (("RightUpperArm", "RightLowerArm"), ("right_shoulder_yaw_link", "right_elbow_link")),
    "right_forearm": (("RightLowerArm", "RightHand"), ("right_elbow_link", "right_rubber_hand_link")),
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


def lateral_axis(left_position, right_position):
    lateral = np.asarray(left_position, dtype=float) - np.asarray(right_position, dtype=float)
    lateral[2] = 0.0
    norm = np.linalg.norm(lateral)
    if norm < 1e-9:
        return np.array([0.0, 1.0, 0.0])
    return lateral / norm


def signed_lateral_clearance(position, center, lateral, side):
    return float(side * np.dot(np.asarray(position) - np.asarray(center), lateral))


def sagittal_lean_degrees(lower_position, upper_position):
    delta = np.asarray(upper_position, dtype=float) - np.asarray(lower_position, dtype=float)
    return float(np.degrees(np.arctan2(delta[0], delta[2])))


def forward_axis_down_pitch_degrees(quat_wxyz):
    forward = np.array([1.0, 0.0, 0.0])
    rotated_forward = R.from_quat(quat_wxyz, scalar_first=True).apply(forward)
    horizontal_norm = np.linalg.norm(rotated_forward[:2])
    return float(np.degrees(np.arctan2(-rotated_forward[2], horizontal_norm)))


def draw_edges(ax, positions, edges, color, label):
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
            linewidth=2.4,
            marker="o",
            markersize=3.5,
            label=label if first else None,
        )
        first = False


def render_comparison_video(args):
    replay = load_iobt_replay_file(
        args.replay_file,
        start=args.start,
        end=args.end,
        required_joint_names=G1_23DOF_REQUIRED_JOINTS,
    )
    if not replay.frames:
        raise SystemExit("Replay selection produced no frames")

    motion_fps = int(round(args.fps if args.fps is not None else replay.fps))
    motion_fps = max(1, motion_fps)
    frames = replay.frames[:: args.stride]
    if args.frame_count is not None:
        frames = frames[: args.frame_count]

    retargeter = GMR(
        src_human=infer_iobt_src_human(replay.metadata),
        tgt_robot=args.robot,
        actual_human_height=replay.actual_human_height,
        verbose=False,
    )
    lower_body_postprocessor = G1IobtLowerBodyPostprocessor(
        retargeter.model,
        knee_flexion_gain=args.knee_flexion_gain,
    )
    ground_offset = compute_ground_offset(retargeter, replay.frames, reference_frame_count=motion_fps)
    retargeter.set_ground_offset(ground_offset)

    robot_body_names = sorted({name for edge in ROBOT_EDGES for name in edge})
    limb_angles = {name: [] for name in LIMB_PAIRS}
    lateral_clearance_errors = {"elbow": [], "hand": []}
    robot_lateral_clearance = {"elbow": [], "hand": []}
    human_torso_leans = []
    robot_torso_leans = []
    human_chest_frame_pitches = []

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.6, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    writer = imageio.get_writer(save_path, fps=motion_fps)
    try:
        for index, frame in enumerate(frames):
            qpos = retargeter.retarget(copy_human_data(frame))
            if not np.all(np.isfinite(qpos)):
                raise ValueError(f"Non-finite qpos at comparison frame {index}")

            qpos = lower_body_postprocessor.apply(qpos, frame)

            qpos, _lowest_support_height, _adjusted_feet = stabilize_g1_support_feet(
                retargeter.model,
                retargeter.configuration.data,
                qpos,
            )
            human_positions = {
                name: np.asarray(pos, dtype=float).copy() + np.array([0.0, args.human_y_offset, 0.0])
                for name, (pos, _quat) in retargeter.scaled_human_data.items()
            }
            human_rotations = {
                name: np.asarray(quat, dtype=float).copy()
                for name, (_pos, quat) in retargeter.scaled_human_data.items()
            }
            robot_positions = body_positions(retargeter.model, retargeter.configuration.data, robot_body_names)

            human_torso_leans.append(sagittal_lean_degrees(human_positions["Hips"], human_positions["Chest"]))
            robot_torso_leans.append(sagittal_lean_degrees(robot_positions["pelvis"], robot_positions["torso_link"]))
            human_chest_frame_pitches.append(forward_axis_down_pitch_degrees(human_rotations["Chest"]))

            human_lateral = lateral_axis(human_positions["LeftShoulder"], human_positions["RightShoulder"])
            robot_lateral = lateral_axis(
                robot_positions["left_shoulder_pitch_link"],
                robot_positions["right_shoulder_pitch_link"],
            )
            clearance_pairs = (
                ("elbow", "LeftLowerArm", "left_elbow_link", 1.0),
                ("hand", "LeftHand", "left_rubber_hand_link", 1.0),
                ("elbow", "RightLowerArm", "right_elbow_link", -1.0),
                ("hand", "RightHand", "right_rubber_hand_link", -1.0),
            )
            for metric_name, human_name, robot_name, side in clearance_pairs:
                human_clearance = signed_lateral_clearance(
                    human_positions[human_name],
                    human_positions["Chest"],
                    human_lateral,
                    side,
                )
                robot_clearance = signed_lateral_clearance(
                    robot_positions[robot_name],
                    robot_positions["torso_link"],
                    robot_lateral,
                    side,
                )
                robot_lateral_clearance[metric_name].append(robot_clearance)
                lateral_clearance_errors[metric_name].append(robot_clearance - human_clearance)

            for limb_name, ((ha, hb), (ra, rb)) in LIMB_PAIRS.items():
                if ha in human_positions and hb in human_positions and ra in robot_positions and rb in robot_positions:
                    limb_angles[limb_name].append(
                        limb_angle_degrees(human_positions[ha], human_positions[hb], robot_positions[ra], robot_positions[rb])
                    )

            root = robot_positions["pelvis"]
            ax.clear()
            ax.view_init(elev=14, azim=-62)
            ax.set_xlim(root[0] - 1.0, root[0] + 1.0)
            ax.set_ylim(root[1] - 1.0, root[1] + 1.0)
            ax.set_zlim(-0.15, 1.9)
            ax.set_xlabel("GMR X forward")
            ax.set_ylabel("GMR Y left")
            ax.set_zlabel("GMR Z up")
            ax.set_title(f"IOBT skeleton vs G1 retarget | frame {index * args.stride}")
            draw_edges(ax, human_positions, HUMAN_EDGES, "#00a6ff", "IOBT skeleton target")
            draw_edges(ax, robot_positions, ROBOT_EDGES, "#ff4b4b", "G1 robot bodies")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.25)

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
    finally:
        writer.close()
        plt.close(fig)

    print(f"Saved comparison video: {save_path}")
    print(f"Frames: {len(frames)}, FPS: {motion_fps}, ground offset: {ground_offset:.4f}")
    for limb_name, values in limb_angles.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        print(
            f"{limb_name}: angle mean={values.mean():.1f} deg, "
            f"p95={np.percentile(values, 95):.1f} deg"
        )
    for metric_name, values in robot_lateral_clearance.items():
        values = np.asarray(values, dtype=float)
        errors = np.asarray(lateral_clearance_errors[metric_name], dtype=float)
        values = values[np.isfinite(values)]
        errors = errors[np.isfinite(errors)]
        if values.size == 0 or errors.size == 0:
            continue
        print(
            f"{metric_name}_lateral_clearance: robot median={np.median(values):.3f} m, "
            f"error median={np.median(errors):+.3f} m"
        )
    print(
        f"torso_sagittal_lean: human median={np.median(human_torso_leans):+.1f} deg, "
        f"robot median={np.median(robot_torso_leans):+.1f} deg"
    )
    print(f"human_chest_frame_pitch_down: median={np.median(human_chest_frame_pitches):+.1f} deg")


def parse_args():
    parser = argparse.ArgumentParser(description="Render an IOBT skeleton vs G1 robot stick-figure comparison video.")
    parser.add_argument("--replay_file", required=True)
    parser.add_argument("--robot", choices=["unitree_g1_23dof"], default="unitree_g1_23dof")
    parser.add_argument("--save_path", default="videos/iobt_mocopi_g1_23dof_skeleton_compare.mp4")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--frame_count", type=int, default=432)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--human_y_offset", type=float, default=0.0)
    parser.add_argument("--knee_flexion_gain", type=float, default=1.10)
    return parser.parse_args()


if __name__ == "__main__":
    render_comparison_video(parse_args())
