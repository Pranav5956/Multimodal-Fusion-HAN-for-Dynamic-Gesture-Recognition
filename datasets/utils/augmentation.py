import numpy as np
import albumentations as A
import cv2

from typing import Tuple

def apply_augmentation(
    skeleton_joints: np.ndarray, 
    depth_features: np.ndarray,
    shift_limit: float = 0.2,
    scale_limit: float = 0.2,
    p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    
        num_frames = skeleton_joints.shape[0]
        
        transform = A.ReplayCompose([
            A.ShiftScaleRotate(
                shift_limit=shift_limit,
                scale_limit=scale_limit,
                rotate_limit=0,
                interpolation=cv2.INTER_LANCZOS4,
                p=p
            )
        ])
        transform_data = transform(image=depth_features[0])
        
        if not transform_data['replay']['transforms'][0]['applied']:
            return skeleton_joints, depth_features

        # applying augmentation
        for i in range(num_frames):
            data = A.ReplayCompose.replay(
                transform_data['replay'], 
                image=depth_features[i]
            )
            depth_features[i] = data["image"]

        params = transform_data['replay']['transforms'][0]['params']

        # shift
        shift = np.array([params['dx'], params['dy'], 0])
        shift[2] = 0
        skeleton_joints = skeleton_joints.reshape(num_frames, -1, 3)
        skeleton_joints = skeleton_joints + shift

        # scale
        scale_factor = params['scale']
        skeleton_joints *= scale_factor

        # time interpolation
        r = np.random.uniform(0, 1)
        for i in range(num_frames - 1):
            skeleton_displacement = skeleton_joints[i + 1] - skeleton_joints[i]
            skeleton_displacement *= r
            skeleton_joints[i + 1, :] = skeleton_joints[i] + \
                    skeleton_displacement

            depth_displacement = depth_features[i + 1] - depth_features[i]
            depth_displacement *= r
            depth_features[i + 1] = depth_features[i] + depth_displacement

        return skeleton_joints, depth_features
