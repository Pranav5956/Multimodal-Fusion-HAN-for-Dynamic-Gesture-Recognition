from typing import Tuple, List
import os
import numpy as np
from glob import glob
import cv2
import processing


Metadata = Tuple[int, int, int, int, int, int, int]


def get_data_list(data_dir: str, split: str = "train") -> List[Metadata]:
    assert split in (
        "train", "test"), f"Valid splits = 'train', 'test': received {split}!"

    data_list = np.loadtxt(os.path.join(
        data_dir, f"{split}_gestures.txt"), dtype=np.uint32).tolist()
    return data_list


def format_data_dir(base_dir: str, metadata: Metadata) -> str:
    gesture_id, finger_id, subject_id, essai_id = metadata[:4]
    return os.path.join(
        base_dir,
        f"gesture_{gesture_id}",
        f"finger_{finger_id}",
        f"subject_{subject_id}",
        f"essai_{essai_id}",
    )


def get_skeleton_joints(data_dir: str) -> np.ndarray:
    filename = "skeletons_world.txt"

    skeleton_joints = np.loadtxt(os.path.join(data_dir, filename))
    skeleton_joints = skeleton_joints.reshape(skeleton_joints.shape[0], -1, 3)

    return skeleton_joints


def get_depth_data(data_dir: str, cropped: bool = False) -> np.ndarray:
    regions = np.loadtxt(os.path.join(data_dir, "general_informations.txt"))
    regions = regions[:, 1:].astype(np.uint32)

    data = []
    for i, filename in enumerate(glob(os.path.join(data_dir, "*_depth.png"))):
        depth_map = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        x, y, w, h = regions[i]

        if cropped:
            depth_map = depth_map[y: y + h, x: x + w]
        else:
            mask = np.zeros_like(depth_map)
            mask[y: y + h, x: x + w] = 1
            depth_map = depth_map * mask

        data.append(depth_map)

    return np.array(data)


def encode_depth_features(depth_map_original, roi: Tuple[int, int, int, int], crop: bool = False) -> np.ndarray:
    mask = np.zeros_like(depth_map_original)
    x, y, w, h = roi
    mask[y: y + h, x: x + w] = 1
    depth_map = depth_map_original * mask

    if crop:
        depth_map = depth_map[y : y + h, x : x + w]
        depth_map = cv2.resize(depth_map, (50, 50))
    else:
        depth_map = cv2.resize(depth_map, (128, 96))

    normal_map = processing.compute_normal_map(depth_map)
    # (x, y, z) -> (r, g, b)
    normal_map = ((normal_map * 0.5 + 0.5) * 255).astype(np.uint8)

    # sharpen features using gvar
    depth_map = processing.grayscale_variation(depth_map).astype(np.uint8)
    depth_map = np.expand_dims(depth_map, axis=-1)

    return np.concatenate([normal_map, depth_map], axis=-1)


def extract_depth_features(depth_features: cv2.Mat) -> Tuple[np.ndarray, np.ndarray]:
    # depth_map, normal_map
    return depth_features[:, :, :3], depth_features[:, :, -1]
