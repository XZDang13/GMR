import mujoco as mj
import numpy as np


G1_FOOT_SUPPORT_POINTS = {
    "left_ankle_roll_link": {
        "heel_outer": np.array([-0.05, 0.025, -0.03]),
        "heel_inner": np.array([-0.05, -0.025, -0.03]),
        "toe_outer": np.array([0.12, 0.03, -0.03]),
        "toe_inner": np.array([0.12, -0.03, -0.03]),
    },
    "right_ankle_roll_link": {
        "heel_outer": np.array([-0.05, 0.025, -0.03]),
        "heel_inner": np.array([-0.05, -0.025, -0.03]),
        "toe_outer": np.array([0.12, 0.03, -0.03]),
        "toe_inner": np.array([0.12, -0.03, -0.03]),
    },
}

G1_ANKLE_PITCH_JOINTS = {
    "left_ankle_roll_link": "left_ankle_pitch_joint",
    "right_ankle_roll_link": "right_ankle_pitch_joint",
}


def _body_support_heights(support_heights, body_name):
    return {
        point_name.split(":", 1)[1]: height
        for point_name, height in support_heights.items()
        if point_name.startswith(f"{body_name}:")
    }


def g1_foot_heel_toe_gap(support_heights, body_name):
    body_heights = _body_support_heights(support_heights, body_name)
    heel_height = np.mean([body_heights["heel_outer"], body_heights["heel_inner"]])
    toe_height = np.mean([body_heights["toe_outer"], body_heights["toe_inner"]])
    return float(heel_height - toe_height)


def g1_foot_min_height(support_heights, body_name):
    return float(min(_body_support_heights(support_heights, body_name).values()))


def g1_foot_support_positions(model, data):
    support_positions = {}
    for body_name, local_points in G1_FOOT_SUPPORT_POINTS.items():
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Robot model is missing body: {body_name}")
        body_pos = data.xpos[body_id].copy()
        body_rot = data.xmat[body_id].reshape(3, 3).copy()
        for point_name, local_point in local_points.items():
            support_positions[f"{body_name}:{point_name}"] = body_pos + body_rot @ local_point
    return support_positions


def g1_foot_support_heights(model, data):
    return {
        name: position[2]
        for name, position in g1_foot_support_positions(model, data).items()
    }


def normalize_g1_qpos_to_ground(model, data, qpos, ground_height=0.0):
    qpos = np.asarray(qpos, dtype=float).copy()
    data.qpos[:] = qpos
    mj.mj_forward(model, data)

    support_heights = g1_foot_support_heights(model, data)
    lowest_support_height = min(support_heights.values())
    qpos[2] += ground_height - lowest_support_height

    data.qpos[:] = qpos
    mj.mj_forward(model, data)
    return qpos, lowest_support_height


def stabilize_g1_support_feet(
    model,
    data,
    qpos,
    ground_height=0.0,
    max_heel_toe_gap=0.04,
    contact_margin=0.03,
    iterations=12,
):
    qpos, _lowest_support_height = normalize_g1_qpos_to_ground(model, data, qpos, ground_height)
    adjusted_feet = 0

    def gap_at(body_name, qpos_candidate):
        data.qpos[:] = qpos_candidate
        mj.mj_forward(model, data)
        return g1_foot_heel_toe_gap(g1_foot_support_heights(model, data), body_name)

    for body_name, joint_name in G1_ANKLE_PITCH_JOINTS.items():
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        support_heights = g1_foot_support_heights(model, data)
        if g1_foot_min_height(support_heights, body_name) > ground_height + contact_margin:
            continue

        current_gap = g1_foot_heel_toe_gap(support_heights, body_name)
        if abs(current_gap) <= max_heel_toe_gap:
            continue

        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Robot model is missing joint: {joint_name}")
        qpos_address = model.jnt_qposadr[joint_id]
        current_value = qpos[qpos_address]
        joint_range = model.jnt_range[joint_id]
        target_gap = np.sign(current_gap) * max_heel_toe_gap

        if current_gap > max_heel_toe_gap:
            low_value = joint_range[0]
            high_value = current_value
        else:
            low_value = current_value
            high_value = joint_range[1]

        low_qpos = qpos.copy()
        low_qpos[qpos_address] = low_value
        high_qpos = qpos.copy()
        high_qpos[qpos_address] = high_value
        low_error = gap_at(body_name, low_qpos) - target_gap
        high_error = gap_at(body_name, high_qpos) - target_gap

        if low_error * high_error > 0.0:
            qpos = low_qpos if abs(low_error) < abs(high_error) else high_qpos
            adjusted_feet += 1
            continue

        for _ in range(iterations):
            mid_value = 0.5 * (low_value + high_value)
            mid_qpos = qpos.copy()
            mid_qpos[qpos_address] = mid_value
            mid_error = gap_at(body_name, mid_qpos) - target_gap
            if low_error * mid_error <= 0.0:
                high_value = mid_value
                high_error = mid_error
            else:
                low_value = mid_value
                low_error = mid_error

        qpos = qpos.copy()
        qpos[qpos_address] = 0.5 * (low_value + high_value)
        adjusted_feet += 1

    qpos, lowest_support_height = normalize_g1_qpos_to_ground(model, data, qpos, ground_height)
    return qpos, lowest_support_height, adjusted_feet
