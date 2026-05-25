import mujoco as mj
import numpy as np


G1_IOBT_KNEE_MAPPINGS = (
    ("LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "left_knee_joint"),
    ("RightUpperLeg", "RightLowerLeg", "RightFoot", "right_knee_joint"),
)


def _joint_qpos_address_and_range(model, joint_name):
    joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Robot model is missing joint: {joint_name}")
    return model.jnt_qposadr[joint_id], model.jnt_range[joint_id].copy()


def _human_knee_flexion_rad(human_data, upper_name, knee_name, foot_name):
    upper_pos = np.asarray(human_data[upper_name][0], dtype=float)
    knee_pos = np.asarray(human_data[knee_name][0], dtype=float)
    foot_pos = np.asarray(human_data[foot_name][0], dtype=float)

    upper_vector = upper_pos - knee_pos
    foot_vector = foot_pos - knee_pos
    norm = np.linalg.norm(upper_vector) * np.linalg.norm(foot_vector)
    if norm < 1e-9:
        return 0.0

    joint_angle = np.arccos(np.clip(np.dot(upper_vector, foot_vector) / norm, -1.0, 1.0))
    return float(max(0.0, np.pi - joint_angle))


class G1IobtLowerBodyPostprocessor:
    def __init__(self, model, knee_flexion_gain=1.0):
        self.knee_flexion_gain = float(knee_flexion_gain)
        self.knee_corrections = []
        for upper_name, knee_name, foot_name, joint_name in G1_IOBT_KNEE_MAPPINGS:
            qpos_address, joint_range = _joint_qpos_address_and_range(model, joint_name)
            self.knee_corrections.append(
                (upper_name, knee_name, foot_name, qpos_address, joint_range)
            )

    @property
    def enabled(self):
        return abs(self.knee_flexion_gain - 1.0) > 1e-6

    def apply(self, qpos, human_data):
        if not self.enabled:
            return qpos

        qpos = np.asarray(qpos, dtype=float).copy()
        for upper_name, knee_name, foot_name, qpos_address, joint_range in self.knee_corrections:
            human_flexion = _human_knee_flexion_rad(human_data, upper_name, knee_name, foot_name)
            qpos[qpos_address] += (self.knee_flexion_gain - 1.0) * human_flexion
            qpos[qpos_address] = np.clip(qpos[qpos_address], joint_range[0], joint_range[1])
        return qpos
