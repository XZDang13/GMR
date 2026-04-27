from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingMatch:
    robot_body_name: str
    human_body_name: str


@dataclass(frozen=True)
class TrackingFilterResult:
    accepted: bool
    threshold: float
    percentile: float
    percentile_error: float
    max_error: float
    num_frames: int
    num_matches: int
    reason: str


def get_position_tracking_matches(retargeter):
    matches = []
    seen = set()

    for match_table in (retargeter.ik_match_table1, retargeter.ik_match_table2):
        for robot_body_name, entry in match_table.items():
            human_body_name, pos_weight = entry[0], entry[1]
            if pos_weight <= 0:
                continue

            key = (robot_body_name, human_body_name)
            if key in seen:
                continue
            seen.add(key)
            matches.append(TrackingMatch(robot_body_name, human_body_name))

    return matches


def compute_position_tracking_errors(retargeter, matches=None):
    import mujoco as mj

    if matches is None:
        matches = get_position_tracking_matches(retargeter)

    mj.mj_forward(retargeter.model, retargeter.configuration.data)
    errors = []

    for match in matches:
        robot_body_id = retargeter.robot_body_names.get(match.robot_body_name)
        if robot_body_id is None:
            errors.append(np.inf)
            continue

        human_target = retargeter.scaled_human_data.get(match.human_body_name)
        if human_target is None:
            errors.append(np.inf)
            continue

        robot_pos = retargeter.configuration.data.xpos[robot_body_id]
        human_pos = np.asarray(human_target[0], dtype=float)
        errors.append(float(np.linalg.norm(robot_pos - human_pos)))

    return np.asarray(errors, dtype=float)


def evaluate_tracking_errors(frame_errors, threshold=0.2, percentile=95):
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    errors = np.asarray(frame_errors, dtype=float)

    if errors.ndim == 1:
        if errors.size == 0:
            num_frames = 0
            num_matches = 0
        else:
            errors = errors[:, None]
            num_frames, num_matches = errors.shape
    elif errors.ndim == 2:
        num_frames, num_matches = errors.shape
    else:
        raise ValueError("frame_errors must be a 1D or 2D array")

    if errors.size == 0:
        return TrackingFilterResult(
            accepted=True,
            threshold=threshold,
            percentile=percentile,
            percentile_error=0.0,
            max_error=0.0,
            num_frames=num_frames,
            num_matches=num_matches,
            reason="no tracking errors",
        )

    if not np.all(np.isfinite(errors)):
        return TrackingFilterResult(
            accepted=False,
            threshold=threshold,
            percentile=percentile,
            percentile_error=np.inf,
            max_error=np.inf,
            num_frames=num_frames,
            num_matches=num_matches,
            reason="non-finite tracking error",
        )

    per_frame_error = np.max(errors, axis=1)
    percentile_error = float(np.percentile(per_frame_error, percentile))
    max_error = float(np.max(per_frame_error))
    accepted = percentile_error <= threshold
    reason = "ok" if accepted else "tracking error above threshold"

    return TrackingFilterResult(
        accepted=accepted,
        threshold=threshold,
        percentile=percentile,
        percentile_error=percentile_error,
        max_error=max_error,
        num_frames=num_frames,
        num_matches=num_matches,
        reason=reason,
    )


def format_tracking_result(result):
    return (
        f"p{result.percentile:g}={result.percentile_error:.3f}m, "
        f"max={result.max_error:.3f}m, "
        f"threshold={result.threshold:.3f}m, "
        f"frames={result.num_frames}, matches={result.num_matches}, "
        f"accepted={result.accepted}"
    )
