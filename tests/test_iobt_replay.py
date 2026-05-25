import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.utils.iobt_replay import (
    CANONICAL_BIND_OFFSET_ENCODING,
    G1_23DOF_REQUIRED_JOINTS,
    OVR_FULL_BODY_CANONICAL_SOURCE,
    OVR_FULL_BODY_MAJOR_JOINTS_SOURCE,
    UNITY_WORLD_COORDINATE_SPACE,
    UNITY_TO_GMR_MATRIX,
    infer_iobt_src_human,
    load_iobt_replay_file,
    unity_quat_xyzw_to_gmr_wxyz,
    unity_position_to_gmr,
)
from general_motion_retargeting.utils.g1_grounding import (
    g1_foot_heel_toe_gap,
    g1_foot_min_height,
    g1_foot_support_heights,
    stabilize_g1_support_feet,
)


RECORDED_REPLAY = Path(
    os.environ.get(
        "IOBT_REPLAY_TEST_FILE",
        "/Users/xdang/Documents/IOBT/Tools/WebRtcSkeletonReceiver/replays/"
        "iobt_skeleton_20260524T115728.010816Z_f548dfb1_one_minute.jsonl",
    )
)


JOINT_PARENTS = {
    "Hips": -1,
    "Spine": 0,
    "Chest": 1,
    "UpperChest": 2,
    "Neck": 3,
    "Head": 4,
    "LeftShoulder": 2,
    "LeftUpperArm": 6,
    "LeftLowerArm": 7,
    "LeftHand": 8,
    "RightShoulder": 2,
    "RightUpperArm": 10,
    "RightLowerArm": 11,
    "RightHand": 12,
    "LeftUpperLeg": 0,
    "LeftLowerLeg": 14,
    "LeftFoot": 15,
    "LeftToes": 16,
    "RightUpperLeg": 0,
    "RightLowerLeg": 18,
    "RightFoot": 19,
    "RightToes": 20,
}


def write_synthetic_replay(path):
    joints = []
    for index, (name, parent) in enumerate(JOINT_PARENTS.items()):
        joints.append(
            {
                "index": index,
                "name": name,
                "parentIndex": parent,
                "bindLocalPosition": {"x": 0.0, "y": 0.1 if parent >= 0 else 0.0, "z": 0.0},
                "bindLocalRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        )

    metadata = {
        "version": 2,
        "source": "SyntheticMocopiCanonical",
        "coordinateSpace": "CanonicalLocal",
        "poseEncoding": CANONICAL_BIND_OFFSET_ENCODING,
        "bindPoseSource": "MocopiAvatarBindPose",
        "skeletonHeightMeters": 1.750136137,
        "skeletonHeightSource": "MocopiAvatarBindPose",
        "joints": joints,
    }
    frame_joints = [
        {
            "position": [1.0, 2.0, 3.0] if index == 0 else [0.0, 0.1, 0.0],
            "rotation": None if index == 1 else [0.0, 0.0, 0.0, 1.0],
        }
        for index in range(len(joints))
    ]
    events = [
        {"type": "replay_start", "format": "iobt-skeleton-replay-jsonl"},
        {"type": "metadata", "metadata": metadata},
        {
            "type": "frame",
            "sequence": 0,
            "timestamp": 1.0,
            "jointCount": len(joints),
            "joints": frame_joints,
        },
        {
            "type": "frame",
            "sequence": 1,
            "timestamp": 1.025,
            "jointCount": len(joints),
            "joints": frame_joints,
        },
        {"type": "replay_end", "frameCount": 2, "metadataCount": 1},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            json.dump(event, handle)
            handle.write("\n")


def write_synthetic_ovr_replay(path):
    joints = [
        {
            "index": index,
            "name": name,
            "parentIndex": parent,
        }
        for index, (name, parent) in enumerate(JOINT_PARENTS.items())
    ]
    metadata = {
        "version": 1,
        "source": OVR_FULL_BODY_MAJOR_JOINTS_SOURCE,
        "coordinateSpace": UNITY_WORLD_COORDINATE_SPACE,
        "skeletonHeightMeters": 1.9825,
        "joints": joints,
    }

    def frame_joint(index):
        name = joints[index]["name"]
        unity_position = [1.0 + index * 0.01, 2.0 + index * 0.02, 3.0 + index * 0.03]
        if name == "Hips":
            unity_position = [1.0, 2.0, 3.0]
        elif name == "Chest":
            unity_position = [1.0, 2.3, 4.0]

        unity_rotation = [0.0, 0.0, 0.0, 1.0]
        if name == "LeftHand":
            unity_rotation = R.from_euler("y", 90.0, degrees=True).as_quat().tolist()
        return {"position": unity_position, "rotation": unity_rotation}

    frame_joints = [frame_joint(index) for index in range(len(joints))]
    events = [
        {"type": "replay_start", "format": "iobt-skeleton-replay-jsonl"},
        {"type": "metadata", "metadata": metadata},
        {
            "type": "frame",
            "sequence": 0,
            "timestamp": 1.0,
            "jointCount": len(joints),
            "joints": frame_joints,
        },
        {
            "type": "frame",
            "sequence": 1,
            "timestamp": 1.025,
            "jointCount": len(joints),
            "joints": frame_joints,
        },
        {"type": "replay_end", "frameCount": 2, "metadataCount": 1},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            json.dump(event, handle)
            handle.write("\n")


def write_synthetic_ovr_canonical_replay(path):
    joints = []
    for index, (name, parent) in enumerate(JOINT_PARENTS.items()):
        bind_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        if name == "Spine":
            bind_position = {"x": 0.0, "y": 0.2, "z": 0.0}
        elif name == "Chest":
            bind_position = {"x": 0.0, "y": 0.3, "z": 0.0}
        elif name == "LeftUpperLeg":
            bind_position = {"x": -0.1, "y": -0.1, "z": 0.0}
        elif name == "RightUpperLeg":
            bind_position = {"x": 0.1, "y": -0.1, "z": 0.0}
        elif parent >= 0:
            bind_position = {"x": 0.0, "y": -0.1, "z": 0.0}

        joints.append(
            {
                "index": index,
                "name": name,
                "parentIndex": parent,
                "bindLocalPosition": bind_position,
                "bindLocalRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        )

    metadata = {
        "version": 2,
        "source": OVR_FULL_BODY_CANONICAL_SOURCE,
        "coordinateSpace": "CanonicalLocal",
        "poseEncoding": CANONICAL_BIND_OFFSET_ENCODING,
        "bindPoseSource": "OVRSkeletonBindPose",
        "skeletonHeightMeters": 1.9825,
        "skeletonHeightSource": "OVRSkeletonBindPose",
        "joints": joints,
    }
    raw_root_rotation = R.from_euler("y", 90.0, degrees=True).as_quat().tolist()
    frame_joints = [
        {
            "position": [1.0, 2.0, 3.0] if index == 0 else [0.0, 0.0, 0.0],
            "rotation": raw_root_rotation if index == 0 else [0.0, 0.0, 0.0, 1.0],
        }
        for index in range(len(joints))
    ]
    events = [
        {"type": "replay_start", "format": "iobt-skeleton-replay-jsonl"},
        {"type": "metadata", "metadata": metadata},
        {
            "type": "frame",
            "sequence": 0,
            "timestamp": 1.0,
            "jointCount": len(joints),
            "joints": frame_joints,
        },
        {
            "type": "frame",
            "sequence": 1,
            "timestamp": 1.025,
            "jointCount": len(joints),
            "joints": frame_joints,
        },
        {"type": "replay_end", "frameCount": 2, "metadataCount": 1},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            json.dump(event, handle)
            handle.write("\n")


def copy_human_data(human_data):
    return {
        name: [np.asarray(pos, dtype=float).copy(), np.asarray(quat, dtype=float).copy()]
        for name, (pos, quat) in human_data.items()
    }


def wrap_degrees(value):
    return (value + 180.0) % 360.0 - 180.0


def heading_from_lateral(left_position, right_position):
    lateral = np.asarray(left_position, dtype=float) - np.asarray(right_position, dtype=float)
    lateral[2] = 0.0
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm < 1e-8:
        return np.nan
    lateral = lateral / lateral_norm
    forward = np.cross(lateral, np.array([0.0, 0.0, 1.0]))
    return np.degrees(np.arctan2(forward[1], forward[0]))


def lateral_axis(left_position, right_position):
    lateral = np.asarray(left_position, dtype=float) - np.asarray(right_position, dtype=float)
    lateral[2] = 0.0
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm < 1e-8:
        return np.array([0.0, 1.0, 0.0])
    return lateral / lateral_norm


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


class IobtReplayTest(unittest.TestCase):
    def test_unity_to_gmr_axis_mapping(self):
        np.testing.assert_allclose(unity_position_to_gmr([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
        np.testing.assert_allclose(unity_position_to_gmr([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(unity_position_to_gmr([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0])

    def test_unity_to_gmr_rotation_mapping(self):
        unity_rotation = R.from_euler("y", 90.0, degrees=True)
        gmr_quat_wxyz = unity_quat_xyzw_to_gmr_wxyz(unity_rotation.as_quat())
        gmr_rotation = R.from_quat(gmr_quat_wxyz, scalar_first=True)

        unity_forward = np.array([0.0, 0.0, 1.0])
        expected = unity_position_to_gmr(unity_rotation.apply(unity_forward))
        actual = gmr_rotation.apply(unity_position_to_gmr(unity_forward))

        np.testing.assert_allclose(
            gmr_rotation.as_matrix(),
            UNITY_TO_GMR_MATRIX @ unity_rotation.as_matrix() @ UNITY_TO_GMR_MATRIX.T,
            atol=1e-12,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_loads_synthetic_canonical_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "synthetic_iobt.jsonl"
            write_synthetic_replay(replay_path)

            replay = load_iobt_replay_file(str(replay_path))

        self.assertEqual(len(replay.metadata["joints"]), 22)
        self.assertEqual(len(replay.frames), 2)
        self.assertTrue(39.9 < replay.fps < 40.1)
        self.assertEqual(replay.actual_human_height, 1.750136137)
        self.assertTrue(set(G1_23DOF_REQUIRED_JOINTS).issubset(replay.frames[0]))

        hips_pos, hips_quat = replay.frames[0]["Hips"]
        chest_pos, chest_quat = replay.frames[0]["Chest"]
        np.testing.assert_allclose(hips_pos, [0.0, 0.0, 2.0], atol=1e-7)
        np.testing.assert_allclose(chest_pos, [0.0, 0.0, 2.2], atol=1e-7)
        self.assertTrue(np.isclose(np.linalg.norm(hips_quat), 1.0))
        self.assertTrue(np.isclose(np.linalg.norm(chest_quat), 1.0))
        self.assertTrue(np.all(np.isfinite(chest_pos)))

    def test_ovr_canonical_replay_uses_ovr_gmr_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "synthetic_ovr_canonical_iobt.jsonl"
            write_synthetic_replay(replay_path)
            lines = replay_path.read_text(encoding="utf-8").splitlines()
            events = [json.loads(line) for line in lines]
            for event in events:
                if event.get("type") == "metadata":
                    event["metadata"]["source"] = OVR_FULL_BODY_CANONICAL_SOURCE
                    event["metadata"]["bindPoseSource"] = "OVRSkeletonBindPose"
                    event["metadata"]["skeletonHeightSource"] = "OVRSkeletonBindPose"
            replay_path.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )

            replay = load_iobt_replay_file(str(replay_path))

        self.assertEqual(infer_iobt_src_human(replay.metadata), "iobt_ovr_fullbody")
        self.assertEqual(replay.metadata["poseEncoding"], CANONICAL_BIND_OFFSET_ENCODING)
        self.assertEqual(replay.metadata["source"], OVR_FULL_BODY_CANONICAL_SOURCE)

    def test_ovr_canonical_root_rotation_uses_body_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "synthetic_ovr_canonical_root_iobt.jsonl"
            write_synthetic_ovr_canonical_replay(replay_path)

            replay = load_iobt_replay_file(str(replay_path), root_position_mode="live")

        hips_quat = replay.frames[0]["Hips"][1]
        hips_rotation = R.from_quat(hips_quat, scalar_first=True)
        forward = hips_rotation.apply([1.0, 0.0, 0.0])
        up = hips_rotation.apply([0.0, 0.0, 1.0])
        lateral = replay.frames[0]["LeftUpperLeg"][0] - replay.frames[0]["RightUpperLeg"][0]
        lateral[2] = 0.0
        lateral = lateral / np.linalg.norm(lateral)
        expected_forward = np.cross(lateral, [0.0, 0.0, 1.0])

        np.testing.assert_allclose(forward, expected_forward, atol=1e-7)
        np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-7)
        self.assertTrue(np.isclose(np.linalg.norm(hips_quat), 1.0))

    def test_loads_synthetic_ovr_unity_world_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "synthetic_ovr_iobt.jsonl"
            write_synthetic_ovr_replay(replay_path)

            replay = load_iobt_replay_file(str(replay_path), root_position_mode="live")

        self.assertEqual(infer_iobt_src_human(replay.metadata), "iobt_ovr_fullbody")
        self.assertEqual(len(replay.metadata["joints"]), 22)
        self.assertEqual(len(replay.frames), 2)
        self.assertEqual(replay.actual_human_height, 1.9825)
        self.assertTrue(set(G1_23DOF_REQUIRED_JOINTS).issubset(replay.frames[0]))

        hips_pos, hips_quat = replay.frames[0]["Hips"]
        chest_pos, chest_quat = replay.frames[0]["Chest"]
        left_hand_quat = replay.frames[0]["LeftHand"][1]
        np.testing.assert_allclose(hips_pos, [3.0, -1.0, 2.0], atol=1e-7)
        np.testing.assert_allclose(chest_pos, [4.0, -1.0, 2.3], atol=1e-7)
        np.testing.assert_allclose(hips_quat, [1.0, 0.0, 0.0, 0.0], atol=1e-7)
        self.assertTrue(np.isclose(np.linalg.norm(chest_quat), 1.0))
        self.assertTrue(np.isclose(np.linalg.norm(left_hand_quat), 1.0))
        self.assertTrue(np.all(np.isfinite(chest_pos)))

    def test_loads_recorded_one_minute_replay_when_available(self):
        if not RECORDED_REPLAY.exists():
            self.skipTest(f"recorded replay not found: {RECORDED_REPLAY}")

        replay = load_iobt_replay_file(str(RECORDED_REPLAY))

        self.assertEqual(len(replay.metadata["joints"]), 22)
        self.assertGreaterEqual(len(replay.frames), 2000)
        self.assertTrue(30.0 < replay.fps < 45.0)
        self.assertEqual(replay.metadata["poseEncoding"], CANONICAL_BIND_OFFSET_ENCODING)
        self.assertTrue(set(G1_23DOF_REQUIRED_JOINTS).issubset(replay.frames[0]))
        for _name, (position, quat) in replay.frames[0].items():
            self.assertTrue(np.all(np.isfinite(position)))
            self.assertTrue(np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5))

    def test_gmr_retargets_recorded_replay_first_frames_when_available(self):
        if not RECORDED_REPLAY.exists():
            self.skipTest(f"recorded replay not found: {RECORDED_REPLAY}")

        import mujoco as mj
        from general_motion_retargeting import GeneralMotionRetargeting as GMR
        from scripts.iobt_replay_to_robot import compute_ground_offset

        replay = load_iobt_replay_file(str(RECORDED_REPLAY), end=30)
        retargeter = GMR(
            src_human="iobt_mocopi",
            tgt_robot="unitree_g1_23dof",
            actual_human_height=replay.actual_human_height,
            verbose=False,
        )
        retargeter.set_ground_offset(
            compute_ground_offset(retargeter, replay.frames, reference_frame_count=round(replay.fps))
        )

        lower_upper_yaw_errors = []
        lowest_support_heights = []
        support_foot_heel_toe_gaps = []
        arm_hand_lateral_clearances = []
        robot_torso_leans = []
        human_chest_frame_pitches = []
        for frame in replay.frames:
            qpos = retargeter.retarget(copy_human_data(frame))
            self.assertEqual(qpos.shape[0], retargeter.model.nq)
            self.assertEqual(qpos[7:].shape[0], retargeter.model.nq - 7)
            self.assertTrue(np.all(np.isfinite(qpos)))

            qpos, _lowest_support_height, _adjusted_feet = stabilize_g1_support_feet(
                retargeter.model,
                retargeter.configuration.data,
                qpos,
            )
            mj.mj_forward(retargeter.model, retargeter.configuration.data)

            model = retargeter.model
            data = retargeter.configuration.data
            support_heights = g1_foot_support_heights(model, data)
            lowest_support_heights.append(min(support_heights.values()))
            support_body_name = min(
                ("left_ankle_roll_link", "right_ankle_roll_link"),
                key=lambda body_name: g1_foot_min_height(support_heights, body_name),
            )
            support_foot_heel_toe_gaps.append(g1_foot_heel_toe_gap(support_heights, support_body_name))

            def robot_body_pos(body_name):
                body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
                return data.xpos[body_id].copy()

            robot_torso_leans.append(
                sagittal_lean_degrees(robot_body_pos("pelvis"), robot_body_pos("torso_link"))
            )
            human_chest_frame_pitches.append(
                forward_axis_down_pitch_degrees(retargeter.scaled_human_data["Chest"][1])
            )
            human_lower_yaw = heading_from_lateral(frame["LeftUpperLeg"][0], frame["RightUpperLeg"][0])
            human_upper_yaw = heading_from_lateral(frame["LeftShoulder"][0], frame["RightShoulder"][0])
            robot_lower_yaw = heading_from_lateral(
                robot_body_pos("left_hip_roll_link"),
                robot_body_pos("right_hip_roll_link"),
            )
            robot_upper_yaw = heading_from_lateral(
                robot_body_pos("left_shoulder_pitch_link"),
                robot_body_pos("right_shoulder_pitch_link"),
            )
            lower_upper_yaw_errors.append(
                wrap_degrees((robot_lower_yaw - robot_upper_yaw) - (human_lower_yaw - human_upper_yaw))
            )

            robot_lateral = lateral_axis(
                robot_body_pos("left_shoulder_pitch_link"),
                robot_body_pos("right_shoulder_pitch_link"),
            )
            arm_hand_lateral_clearances.extend(
                [
                    signed_lateral_clearance(
                        robot_body_pos("left_rubber_hand_link"),
                        robot_body_pos("torso_link"),
                        robot_lateral,
                        1.0,
                    ),
                    signed_lateral_clearance(
                        robot_body_pos("right_rubber_hand_link"),
                        robot_body_pos("torso_link"),
                        robot_lateral,
                        -1.0,
                    ),
                ]
            )

        self.assertLess(np.percentile(np.abs(lower_upper_yaw_errors), 95), 25.0)
        self.assertLess(np.max(np.abs(lowest_support_heights)), 1e-8)
        self.assertLess(np.percentile(np.abs(support_foot_heel_toe_gaps), 95), 0.05)
        self.assertGreater(np.median(arm_hand_lateral_clearances), 0.15)
        self.assertLess(np.median(robot_torso_leans), 4.0)
        self.assertLess(np.median(human_chest_frame_pitches), 10.0)


if __name__ == "__main__":
    unittest.main()
