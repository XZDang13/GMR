import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.iobt_utils import (
    CANONICAL_BIND_OFFSET_ENCODING,
    DEFAULT_CANONICAL_HUMAN_HEIGHT,
    IOBTSkeletonSource,
    add_hand_roll_targets,
    unity_position_to_gmr,
    unity_rotation_to_gmr,
)


JOINTS = [
    ("Hips", -1, [0.0, 0.0, 0.0]),
    ("Spine", 0, [0.0, 0.1, 0.0]),
    ("Chest", 1, [0.0, 0.2, 0.0]),
    ("UpperChest", 2, [0.0, 0.15, 0.0]),
    ("Neck", 3, [0.0, 0.12, 0.0]),
    ("Head", 4, [0.0, 0.1, 0.0]),
    ("LeftShoulder", 2, [-0.1, 0.1, 0.0]),
    ("LeftUpperArm", 6, [-0.2, 0.0, 0.0]),
    ("LeftLowerArm", 7, [-0.25, 0.0, 0.0]),
    ("LeftHand", 8, [-0.2, 0.0, 0.0]),
    ("RightShoulder", 2, [0.1, 0.1, 0.0]),
    ("RightUpperArm", 10, [0.2, 0.0, 0.0]),
    ("RightLowerArm", 11, [0.25, 0.0, 0.0]),
    ("RightHand", 12, [0.2, 0.0, 0.0]),
    ("LeftUpperLeg", 0, [-0.1, -0.1, 0.0]),
    ("LeftLowerLeg", 14, [0.0, -0.4, 0.0]),
    ("LeftFoot", 15, [0.0, -0.4, 0.0]),
    ("LeftToes", 16, [0.0, 0.0, 0.15]),
    ("RightUpperLeg", 0, [0.1, -0.1, 0.0]),
    ("RightLowerLeg", 18, [0.0, -0.4, 0.0]),
    ("RightFoot", 19, [0.0, -0.4, 0.0]),
    ("RightToes", 20, [0.0, 0.0, 0.15]),
]


def write_synthetic_replay(
    path: Path,
    skeleton_height_meters: float = DEFAULT_CANONICAL_HUMAN_HEIGHT,
    skeleton_height_source: str = "SyntheticBindPose",
) -> None:
    metadata = {
        "version": 2,
        "source": "SyntheticIOBTCanonical",
        "coordinateSpace": "CanonicalLocal",
        "poseEncoding": CANONICAL_BIND_OFFSET_ENCODING,
        "bindPoseSource": "SyntheticBindPose",
        "skeletonHeightMeters": skeleton_height_meters,
        "skeletonHeightSource": skeleton_height_source,
        "joints": [
            {
                "index": index,
                "id": index,
                "name": name,
                "parentIndex": parent,
                "bindLocalPosition": {"x": offset[0], "y": offset[1], "z": offset[2]},
                "bindLocalRotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
            for index, (name, parent, offset) in enumerate(JOINTS)
        ],
    }

    def frame(sequence, root_position, chest_rotation=None):
        joints = []
        for index, _joint in enumerate(JOINTS):
            rotation = [0.0, 0.0, 0.0, 1.0]
            if index == 2 and chest_rotation is not None:
                rotation = chest_rotation
            joints.append(
                {
                    "position": root_position if index == 0 else [99.0, 99.0, 99.0],
                    "rotation": rotation,
                }
            )
        return {
            "type": "frame",
            "sequence": sequence,
            "timestamp": sequence / 50.0,
            "receivedAtUnix": 1000.0 + sequence / 50.0,
            "joints": joints,
        }

    events = [
        {"type": "replay_start", "format": "iobt-skeleton-replay-jsonl"},
        {"type": "metadata", "metadata": metadata},
        frame(0, [1.0, 2.0, 3.0]),
        frame(1, [1.5, 2.0, 3.0], R.from_euler("z", 30, degrees=True).as_quat().tolist()),
        {"type": "replay_end", "frameCount": 2, "metadataCount": 1},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


class IOBTCanonicalAdapterTest(unittest.TestCase):
    def test_coordinate_conversion_axes(self):
        np.testing.assert_allclose(unity_position_to_gmr([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
        np.testing.assert_allclose(unity_position_to_gmr([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(unity_position_to_gmr([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0])
        np.testing.assert_allclose(np.linalg.norm(unity_rotation_to_gmr([0.0, 0.0, 0.0, 1.0])), 1.0)

    def test_replay_and_live_use_same_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "synthetic_iobt.jsonl"
            write_synthetic_replay(replay_path)

            replay_source = IOBTSkeletonSource(source="replay", input_path=str(replay_path))
            live_source = IOBTSkeletonSource(
                source="live",
                input_path=str(replay_path),
                stop_on_replay_end=True,
            )
            live_frames = list(live_source.iter_frames())

        self.assertEqual(len(replay_source.frames), 2)
        self.assertEqual(len(live_frames), 2)
        self.assertEqual(set(replay_source.frames[0].human_data), set(live_frames[0].human_data))
        for joint_name in replay_source.frames[0].human_data:
            np.testing.assert_allclose(
                replay_source.frames[0].human_data[joint_name][0],
                live_frames[0].human_data[joint_name][0],
            )
            np.testing.assert_allclose(
                replay_source.frames[0].human_data[joint_name][1],
                live_frames[0].human_data[joint_name][1],
            )

    def test_bind_offsets_define_segment_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "synthetic_iobt.jsonl"
            write_synthetic_replay(replay_path)
            source = IOBTSkeletonSource(source="replay", input_path=str(replay_path))

        for parent_name, child_name in (("LeftLowerLeg", "LeftFoot"), ("LeftFoot", "LeftToes")):
            first = source.frames[0].human_data
            second = source.frames[1].human_data
            first_length = np.linalg.norm(first[child_name][0] - first[parent_name][0])
            second_length = np.linalg.norm(second[child_name][0] - second[parent_name][0])
            np.testing.assert_allclose(first_length, second_length, atol=1e-8)

    def test_smplx_body_only_height_uses_canonical_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "synthetic_iobt.jsonl"
            write_synthetic_replay(
                replay_path,
                skeleton_height_meters=1.66,
                skeleton_height_source="SMPLXBodyOnlyBindPose",
            )
            source = IOBTSkeletonSource(source="replay", input_path=str(replay_path))

        self.assertEqual(source.actual_human_height, DEFAULT_CANONICAL_HUMAN_HEIGHT)

    def test_noncanonical_height_source_keeps_reported_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "synthetic_iobt.jsonl"
            write_synthetic_replay(
                replay_path,
                skeleton_height_meters=1.66,
                skeleton_height_source="MeasuredRuntimeSkeleton",
            )
            source = IOBTSkeletonSource(source="replay", input_path=str(replay_path))

        self.assertEqual(source.actual_human_height, 1.66)

    def test_hand_roll_target_preserves_palm_roll_without_full_wrist_target(self):
        human_data = {
            "LeftLowerArm": [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])],
            "LeftHand": [np.array([1.0, 0.0, 0.0]), R.identity().as_quat(scalar_first=True)],
            "RightLowerArm": [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])],
            "RightHand": [
                np.array([1.0, 0.0, 0.0]),
                R.from_euler("x", 90, degrees=True).as_quat(scalar_first=True),
            ],
        }
        add_hand_roll_targets(human_data)

        left_roll = R.from_quat(human_data["LeftHandRoll"][1], scalar_first=True)
        right_roll = R.from_quat(human_data["RightHandRoll"][1], scalar_first=True)
        np.testing.assert_allclose(left_roll.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(right_roll.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-8)
        self.assertGreater((left_roll.inv() * right_roll).magnitude(), 1.0)

    def test_hand_roll_target_applies_g1_wrist_roll_offsets(self):
        human_data = {
            "LeftLowerArm": [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])],
            "LeftHand": [
                np.array([1.0, 0.0, 0.0]),
                R.from_euler("z", -90, degrees=True).as_quat(scalar_first=True),
            ],
            "RightLowerArm": [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])],
            "RightHand": [
                np.array([1.0, 0.0, 0.0]),
                R.from_euler("z", 90, degrees=True).as_quat(scalar_first=True),
            ],
        }
        add_hand_roll_targets(human_data)

        left_roll = R.from_quat(human_data["LeftHandRoll"][1], scalar_first=True)
        right_roll = R.from_quat(human_data["RightHandRoll"][1], scalar_first=True)
        np.testing.assert_allclose(left_roll.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(right_roll.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(left_roll.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-8)
        np.testing.assert_allclose(right_roll.apply([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-8)

    def test_hand_roll_target_uses_side_specific_forearm_axis_fallback(self):
        left_data = {
            "LeftLowerArm": [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])],
            "LeftHand": [np.array([0.0, 0.0, 0.0]), R.identity().as_quat(scalar_first=True)],
        }
        right_data = {
            "RightLowerArm": [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])],
            "RightHand": [np.array([0.0, 0.0, 0.0]), R.identity().as_quat(scalar_first=True)],
        }
        add_hand_roll_targets(left_data)
        add_hand_roll_targets(right_data)

        left_roll = R.from_quat(left_data["LeftHandRoll"][1], scalar_first=True)
        right_roll = R.from_quat(right_data["RightHandRoll"][1], scalar_first=True)
        np.testing.assert_allclose(left_roll.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-8)
        np.testing.assert_allclose(right_roll.apply([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-8)

    def test_gmr_can_initialize_and_retarget_synthetic_frames(self):
        try:
            from general_motion_retargeting import GeneralMotionRetargeting
            import mujoco as mj
        except ModuleNotFoundError as exc:
            self.skipTest(f"GMR runtime dependency is not installed: {exc.name}")

        with tempfile.TemporaryDirectory() as tmp:
            replay_path = Path(tmp) / "synthetic_iobt.jsonl"
            write_synthetic_replay(replay_path)
            source = IOBTSkeletonSource(source="replay", input_path=str(replay_path))

        retargeter = GeneralMotionRetargeting(
            src_human="iobt_canonical",
            tgt_robot="unitree_g1_23dof",
            actual_human_height=source.actual_human_height,
            verbose=False,
            max_iter=1,
        )
        qpos = retargeter.retarget(source.frames[0].human_data)
        self.assertEqual(qpos.shape, (retargeter.model.nq,))
        self.assertTrue(np.all(np.isfinite(qpos)))
        self.assertIsNotNone(retargeter.posture_task)

        for joint_name, expected_cost in (
            ("left_ankle_roll_joint", 2.0),
            ("right_ankle_roll_joint", 2.0),
            ("left_elbow_joint", 0.5),
            ("right_elbow_joint", 0.5),
            ("left_shoulder_roll_joint", 0.4),
            ("right_shoulder_roll_joint", 0.4),
            ("left_shoulder_yaw_joint", 0.8),
            ("right_shoulder_yaw_joint", 0.8),
        ):
            joint_id = mj.mj_name2id(retargeter.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            self.assertGreaterEqual(joint_id, 0)
            dof_index = retargeter.model.jnt_dofadr[joint_id]
            self.assertEqual(retargeter.posture_task.cost[dof_index], expected_cost)

        left_hand_task = retargeter.human_body_to_task2["LeftHandRoll"]
        right_hand_task = retargeter.human_body_to_task2["RightHandRoll"]
        np.testing.assert_allclose(left_hand_task.cost[:3], [12.0, 12.0, 12.0])
        np.testing.assert_allclose(right_hand_task.cost[:3], [12.0, 12.0, 12.0])
        np.testing.assert_allclose(left_hand_task.cost[3:], [0.1, 0.0, 0.0])
        np.testing.assert_allclose(right_hand_task.cost[3:], [0.1, 0.0, 0.0])

    def test_optional_anchor_scaling_keeps_limb_scale_relative_to_anchor(self):
        try:
            from general_motion_retargeting import GeneralMotionRetargeting
        except ModuleNotFoundError as exc:
            self.skipTest(f"GMR runtime dependency is not installed: {exc.name}")

        retargeter = GeneralMotionRetargeting(
            src_human="iobt_canonical",
            tgt_robot="unitree_g1_23dof",
            verbose=False,
            max_iter=1,
        )
        human_data = {
            "Hips": [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])],
            "Chest": [np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])],
            "LeftHand": [np.array([3.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])],
        }
        scale_table = {"Hips": 1.0, "Chest": 1.0, "LeftHand": 0.5}

        root_scaled = retargeter.scale_human_data(
            {key: [value[0].copy(), value[1].copy()] for key, value in human_data.items()},
            "Hips",
            scale_table,
        )
        anchor_scaled = retargeter.scale_human_data(
            {key: [value[0].copy(), value[1].copy()] for key, value in human_data.items()},
            "Hips",
            scale_table,
            {"LeftHand": "Chest"},
        )

        np.testing.assert_allclose(root_scaled["LeftHand"][0], [2.0, 0.0, 0.5])
        np.testing.assert_allclose(anchor_scaled["LeftHand"][0], [2.0, 0.0, 1.0])

    def test_g1_config_does_not_reapply_source_orientation_offsets(self):
        config_path = (
            Path(__file__).parents[1]
            / "general_motion_retargeting"
            / "ik_configs"
            / "iobt_canonical_to_g1_23dof.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["human_scale_table"]["LeftShoulder"], 0.7)
        self.assertEqual(config["human_scale_table"]["RightShoulder"], 0.7)
        self.assertEqual(config["human_scale_table"]["LeftShoulderSocket"], 0.9)
        self.assertEqual(config["human_scale_table"]["RightShoulderSocket"], 0.9)
        self.assertNotIn("LeftLowerArm", config["root_frame_position_offsets"])
        self.assertNotIn("RightLowerArm", config["root_frame_position_offsets"])
        np.testing.assert_allclose(config["root_frame_position_offsets"]["LeftFoot"], [-0.08, 0.05, -0.03])
        np.testing.assert_allclose(config["root_frame_position_offsets"]["LeftToes"], [-0.08, 0.05, 0.0])
        np.testing.assert_allclose(config["root_frame_position_offsets"]["RightFoot"], [-0.08, -0.05, -0.03])
        np.testing.assert_allclose(config["root_frame_position_offsets"]["RightToes"], [-0.08, -0.05, 0.0])
        self.assertEqual(config["joint_regularization"]["left_ankle_roll_joint"], 2.0)
        self.assertEqual(config["joint_regularization"]["right_ankle_roll_joint"], 2.0)
        self.assertEqual(config["joint_regularization"]["left_elbow_joint"], 0.5)
        self.assertEqual(config["joint_regularization"]["right_elbow_joint"], 0.5)
        self.assertEqual(config["joint_regularization"]["left_shoulder_roll_joint"], 0.4)
        self.assertEqual(config["joint_regularization"]["right_shoulder_roll_joint"], 0.4)
        self.assertEqual(config["joint_regularization"]["left_shoulder_yaw_joint"], 0.8)
        self.assertEqual(config["joint_regularization"]["right_shoulder_yaw_joint"], 0.8)
        self.assertNotIn("left_knee_joint", config["joint_regularization"])
        self.assertNotIn("right_knee_joint", config["joint_regularization"])
        for body_name in (
            "LeftUpperLeg",
            "LeftLowerLeg",
            "LeftFoot",
            "LeftToes",
            "RightUpperLeg",
            "RightLowerLeg",
            "RightFoot",
            "RightToes",
        ):
            self.assertEqual(config["human_scale_table"][body_name], 0.86)
        for body_name in (
            "LeftUpperArm",
            "LeftLowerArm",
            "LeftHand",
            "LeftHandRoll",
            "RightUpperArm",
            "RightLowerArm",
            "RightHand",
            "RightHandRoll",
        ):
            self.assertEqual(config["human_scale_table"][body_name], 0.72)

        self.assertEqual(config["ik_match_table2"]["pelvis"][1], 12)
        self.assertEqual(config["ik_match_table2"]["left_knee_link"][1], 18)
        self.assertEqual(config["ik_match_table2"]["right_knee_link"][1], 18)
        self.assertEqual(config["ik_match_table2"]["left_ankle_roll_link"][1], 36)
        self.assertEqual(config["ik_match_table2"]["right_ankle_roll_link"][1], 36)
        self.assertEqual(config["ik_match_table2"]["left_toe_link"][1], 80)
        self.assertEqual(config["ik_match_table2"]["right_toe_link"][1], 80)

        anchor_table = config["human_scale_anchor_table"]
        self.assertEqual(anchor_table["LeftShoulder"], "Chest")
        self.assertEqual(anchor_table["LeftUpperArm"], "LeftShoulderSocket")
        self.assertEqual(anchor_table["LeftLowerArm"], "LeftUpperArm")
        self.assertEqual(anchor_table["LeftHandRoll"], "LeftLowerArm")
        self.assertEqual(anchor_table["RightShoulder"], "Chest")
        self.assertEqual(anchor_table["RightUpperArm"], "RightShoulderSocket")
        self.assertEqual(anchor_table["RightLowerArm"], "RightUpperArm")
        self.assertEqual(anchor_table["RightHandRoll"], "RightLowerArm")

        for table_name in ("ik_match_table1", "ik_match_table2"):
            for _robot_body, entry in config[table_name].items():
                np.testing.assert_allclose(entry[4], [1.0, 0.0, 0.0, 0.0])

        for table_name in ("ik_match_table1", "ik_match_table2"):
            table = config[table_name]
            self.assertEqual(table["pelvis"][0], "Hips")
            self.assertEqual(table["pelvis"][2], 4)
            self.assertEqual(table["torso_link"][0], "Chest")
            self.assertEqual(table["torso_link"][1], 0)
            self.assertEqual(table["torso_link"][2], 6)
            self.assertEqual(table["left_shoulder_yaw_link"][0], "LeftShoulderSocket")
            self.assertEqual(table["right_shoulder_yaw_link"][0], "RightShoulderSocket")
            self.assertEqual(table["left_shoulder_yaw_link"][1], 0)
            self.assertEqual(table["right_shoulder_yaw_link"][1], 0)
            self.assertEqual(table["left_rubber_hand_link"][0], "LeftHandRoll")
            self.assertEqual(table["right_rubber_hand_link"][0], "RightHandRoll")
            left_roll_cost = np.atleast_1d(table["left_rubber_hand_link"][2]).astype(float)
            right_roll_cost = np.atleast_1d(table["right_rubber_hand_link"][2]).astype(float)
            if table_name == "ik_match_table1":
                np.testing.assert_allclose(left_roll_cost, [0.0])
                np.testing.assert_allclose(right_roll_cost, [0.0])
            else:
                np.testing.assert_allclose(left_roll_cost, [0.1, 0.0, 0.0])
                np.testing.assert_allclose(right_roll_cost, [0.1, 0.0, 0.0])
                self.assertEqual(table["left_elbow_link"][1], 12)
                self.assertEqual(table["right_elbow_link"][1], 12)
                self.assertEqual(table["left_rubber_hand_link"][1], 12)
                self.assertEqual(table["right_rubber_hand_link"][1], 12)
                self.assertEqual(table["left_knee_link"][1], 18)
                self.assertEqual(table["right_knee_link"][1], 18)
            for robot_body in (
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "right_shoulder_yaw_link",
                "right_elbow_link",
            ):
                self.assertEqual(table[robot_body][2], 0)


if __name__ == "__main__":
    unittest.main()
