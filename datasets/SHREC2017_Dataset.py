from torch.utils.data import Dataset
from .utils.augmentation import apply_augmentation
from .utils.sampling import sample_from_frames
import numpy as np
import cv2
import torch
import os

from typing import Dict, Tuple, List


class SHREC2017_Dataset(Dataset):
    def __init__(self, num_classes: int, config: Dict, split: str = "train", cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}) -> None:
        super(SHREC2017_Dataset, self).__init__()
        
        self._num_classes = num_classes
        self._split = split
        
        self._data_dir = config["data_dir"]
        self._images_variant = config["images"]
        self._skeletons_variant = config["skeletons"]
        self._sampling_strategy = config["sampling"]["strategy"]
        self._sampling_size = config["sampling"]["size"]
        
        self._apply_augmentations = config["augmentation"]["apply"]
        self._augmentation_transforms = config["augmentation"]["transforms"]

        assert self._split in ("train", "test"), f"Valid splits = 'train', 'test': received '{self._split}'!"
        assert self._num_classes in (
            14, 28), f"Valid num_classes = 14, 28: received {self._num_classes}!"
        assert self._images_variant in (
            "cropped", "uncropped"), f"Valid dataset variants = 'cropped', 'uncropped': received {self._images_variant}!"
        assert self._skeletons_variant in (
            "2d", "3d"), f"Valid dataset variants = '2d', '3d': received {self._skeletons_variant}!"
        assert self._sampling_strategy in (
            "uniform", "k-random"), f"Valid sampling = 'k-random', 'uniform': received {self._sampling_strategy}!"
        
        # Load metadata
        self._metadata = np.loadtxt(os.path.join(self._data_dir, self._split, "metadata.txt"), dtype=np.uint8)
        self._cache = cache
    
    def _get_data(self, index: int, sample_frames: List[int]) -> np.ndarray:
        root_dir = os.path.join(self._data_dir, self._split, str(index))
        images_file_path = [os.path.join(root_dir, self._images_variant, f"{frame}.png") for frame in sample_frames]
        skeleton_file_path = os.path.join(root_dir, f"skeletons_{self._skeletons_variant}.txt")
        
        skeleton_joints = np.loadtxt(skeleton_file_path, dtype=np.float32)
        skeleton_joints = skeleton_joints.reshape(
            skeleton_joints.shape[0], 22, -1)[sample_frames]
        
        depth_features = []
        for image_filename in images_file_path:
            depth_feature = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth_features.append(depth_feature)
        depth_features = np.stack(depth_features, axis=0)
        
        return skeleton_joints, depth_features
        
    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        metadata = self._metadata[index]
        label = metadata[4 if self._num_classes == 14 else 5] - 1
        num_frames = metadata[6]
        
        # sampling
        sample_frames = sample_from_frames(num_frames, self._sampling_size, self._sampling_strategy)        
        skeleton_joints, depth_features = self._get_data(index, sample_frames)
        assert skeleton_joints.shape[0] == depth_features.shape[0] == len(sample_frames), f"Samples count mismatch!"
        
        # augmentation
        if self._split == "train" and self._apply_augmentations:
            skeleton_joints, depth_features = apply_augmentation(
                skeleton_joints,
                depth_features,
                **self._augmentation_transforms
            )
        
        # feature processing
        skeleton_joints1 = skeleton_joints - skeleton_joints[0, 1, :]
        skeleton_joints2 = skeleton_joints - skeleton_joints[0]
        skeleton_joints = np.concatenate(
            (skeleton_joints1, skeleton_joints2), axis=-1)
        depth_features = (depth_features / 255)
        
        return (
            torch.from_numpy(skeleton_joints.astype(np.float32)),
            torch.from_numpy(depth_features.astype(np.float32)),
            torch.tensor(label, dtype=torch.uint8)
        )
    
    def __len__(self) -> int:
        return self._metadata.shape[0]   
        