import argparse
import os
import pathlib
import pickle
import time

import numpy as np
import torch
from natsort import natsorted
from rich import print
from tqdm import tqdm

from general_motion_retargeting import (
    GeneralMotionRetargeting as GMR,
    IK_CONFIG_DICT,
    ROBOT_XML_DICT,
    RobotMotionViewer,
)
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


HERE = pathlib.Path(__file__).parent
SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"


class ReviewControls:
    def __init__(self):
        self.paused = False
        self.replay = False
        self.decision = None

    def keyboard_callback(self, keycode):
        if keycode in (27, 256):
            self.decision = "quit"
            return

        if keycode == ord(" "):
            self.paused = not self.paused
            return

        if not 0 <= keycode < 256:
            return

        key = chr(keycode).lower()
        if key == "s":
            self.decision = "save"
        elif key == "d":
            self.decision = "drop"
        elif key == "r":
            self.replay = True
        elif key == "q":
            self.decision = "quit"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively review SMPL-X motions after GMR retargeting."
    )
    parser.add_argument(
        "--src_folder",
        required=True,
        type=pathlib.Path,
        help="Folder containing SMPL-X .npz/.pkl motion files.",
    )
    parser.add_argument(
        "--tgt_folder",
        required=True,
        type=pathlib.Path,
        help="Folder where accepted robot-motion .pkl files will be saved.",
    )
    parser.add_argument(
        "--robot",
        choices=natsorted(IK_CONFIG_DICT["smplx"].keys()),
        default="unitree_g1",
        help="Target robot name.",
    )
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
        help="Target FPS used when converting SMPL-X frames.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Review motions even when the accepted output already exists.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
        help="Preview at the retargeted motion FPS instead of as fast as possible.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for kinematics metadata: auto, cpu, cuda, cuda:0, etc.",
    )
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")
    return device


def discover_motion_files(src_folder):
    motion_files = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in filenames:
            path = pathlib.Path(dirpath) / filename
            if path.name.endswith("_stagei.npz"):
                continue
            if path.suffix in (".npz", ".pkl"):
                motion_files.append(path)
    return natsorted(motion_files, key=lambda path: str(path))


def output_path_for_motion(src_file, src_folder, tgt_folder):
    relative_path = src_file.relative_to(src_folder).with_suffix(".pkl")
    return tgt_folder / relative_path


def assert_safe_folders(src_folder, tgt_folder):
    if not src_folder.exists():
        raise FileNotFoundError(f"Source folder does not exist: {src_folder}")
    if not src_folder.is_dir():
        raise NotADirectoryError(f"Source path is not a folder: {src_folder}")
    src_resolved = src_folder.resolve()
    tgt_resolved = tgt_folder.resolve()
    if tgt_resolved == src_resolved or tgt_resolved.is_relative_to(src_resolved):
        raise ValueError(
            "--tgt_folder must be outside --src_folder. SMPL-X .pkl inputs "
            "and robot .pkl outputs use the same extension."
        )


def load_and_retarget_motion(smplx_file, robot, target_fps):
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        smplx_file, SMPLX_FOLDER
    )
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=target_fps,
    )

    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
        verbose=False,
    )

    qpos_list = []
    for smplx_frame_data in tqdm(smplx_data_frames, desc="Retargeting", leave=False):
        qpos = retargeter.retarget(smplx_frame_data)
        qpos_list.append(qpos.copy())

    if not qpos_list:
        raise ValueError("Motion contains no frames after SMPL-X conversion.")

    return np.asarray(qpos_list), aligned_fps


def build_motion_data(qpos_list, aligned_fps, kinematics_model, device):
    root_pos = qpos_list[:, :3].copy()
    root_rot = qpos_list[:, 3:7][:, [1, 2, 3, 0]].copy()
    dof_pos = qpos_list[:, 7:].copy()

    num_frames = root_pos.shape[0]
    identity_root_pos = torch.zeros((num_frames, 3), device=device)
    identity_root_rot = torch.zeros((num_frames, 4), device=device)
    identity_root_rot[:, -1] = 1.0

    with torch.no_grad():
        local_body_pos, _ = kinematics_model.forward_kinematics(
            identity_root_pos,
            identity_root_rot,
            torch.from_numpy(dof_pos).to(device=device, dtype=torch.float),
        )

    return {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": kinematics_model.body_names,
    }


def save_motion_data(motion_data, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(motion_data, f)


def viewer_is_running(viewer):
    is_running = getattr(viewer.viewer, "is_running", None)
    if is_running is None:
        return True
    return is_running()


def close_viewer(viewer):
    try:
        viewer.close()
    except Exception as exc:
        print(f"[warning] Error while closing viewer: {exc}")


def review_motion(motion_name, motion_data, robot, rate_limit):
    controls = ReviewControls()
    viewer = RobotMotionViewer(
        robot_type=robot,
        motion_fps=motion_data["fps"],
        transparent_robot=0,
        keyboard_callback=controls.keyboard_callback,
    )

    root_rot_wxyz = motion_data["root_rot"][:, [3, 0, 1, 2]]
    frame_idx = 0

    print(f"[bold]Reviewing[/bold] {motion_name}")
    print("Viewer keys: [S] save, [D] drop, [R] replay, [Space] pause, [Q/Esc] quit")

    try:
        while controls.decision is None:
            if not viewer_is_running(viewer):
                controls.decision = "quit"
                break

            if controls.replay:
                frame_idx = 0
                controls.replay = False

            viewer.step(
                motion_data["root_pos"][frame_idx],
                root_rot_wxyz[frame_idx],
                motion_data["dof_pos"][frame_idx],
                rate_limit=rate_limit,
                follow_camera=True,
            )

            if not controls.paused:
                frame_idx = (frame_idx + 1) % len(motion_data["root_pos"])
            elif not rate_limit:
                time.sleep(0.02)
    finally:
        close_viewer(viewer)

    return controls.decision


def main():
    args = parse_args()
    args.src_folder = args.src_folder.expanduser()
    args.tgt_folder = args.tgt_folder.expanduser()
    assert_safe_folders(args.src_folder, args.tgt_folder)

    device = resolve_device(args.device)
    motion_files = discover_motion_files(args.src_folder)
    print(f"Found {len(motion_files)} SMPL-X motion files in {args.src_folder}")
    print(f"Accepted motions will be saved under {args.tgt_folder}")
    print(f"Using kinematics device: {device}")

    if not motion_files:
        return

    kinematics_model = KinematicsModel(str(ROBOT_XML_DICT[args.robot]), device=device)

    processed = 0
    saved = 0
    dropped = 0
    skipped = 0
    failed = 0

    for idx, smplx_file in enumerate(motion_files, start=1):
        target_path = output_path_for_motion(smplx_file, args.src_folder, args.tgt_folder)

        if target_path.exists() and not args.override:
            skipped += 1
            print(f"[{idx}/{len(motion_files)}] Skipping existing output: {target_path}")
            continue

        print(f"[{idx}/{len(motion_files)}] Processing {smplx_file}")
        try:
            qpos_list, aligned_fps = load_and_retarget_motion(
                smplx_file,
                args.robot,
                args.target_fps,
            )
            motion_data = build_motion_data(
                qpos_list,
                aligned_fps,
                kinematics_model,
                device,
            )
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[error] Failed to process {smplx_file}: {exc}")
            continue

        decision = review_motion(
            str(smplx_file.relative_to(args.src_folder)),
            motion_data,
            args.robot,
            args.rate_limit,
        )

        if decision == "save":
            save_motion_data(motion_data, target_path)
            saved += 1
            print(f"Saved: {target_path}")
        elif decision == "drop":
            dropped += 1
            print(f"Dropped: {smplx_file}")
        elif decision == "quit":
            print("Review stopped by user.")
            break

    print(
        "Done. "
        f"processed={processed}, saved={saved}, dropped={dropped}, "
        f"skipped={skipped}, failed={failed}"
    )


if __name__ == "__main__":
    main()
