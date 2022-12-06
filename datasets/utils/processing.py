import cv2
import numpy as np

def compute_normal_map(depth_map, kernel_size: int = 3, epsilon: float = 1e-12):
    depth_map = depth_map.astype(np.float32)
    
    zx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=kernel_size)
    zy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=kernel_size)

    normal_map = np.dstack((-zx, -zy, np.ones_like(depth_map)))

    scale = np.linalg.norm(normal_map, axis=-1) + epsilon
    normal_map[:, :, 0] /= scale
    normal_map[:, :, 1] /= scale
    normal_map[:, :, 2] /= scale

    return normal_map


def grayscale_variation(image: np.ndarray, eta: int = 10, g_min: int = 155, g_max: int = 255, near_depth_thresh: int = 0, far_depth_thresh: int = None) -> np.ndarray:
    """Quantizes depth levels into discrete grayscale image levels.
    Args:
        image (np.ndarray): Input image of shape (H, W).
        eta (int, optional): Number of gray levels. Defaults to 10.
        g_min (int, optional): Lowest gray level. Defaults to 155.
        g_max (int, optional): Highest gray level. Defaults to 255.
        near_depth_thresh (int, optional): Minimum considered depth. Defaults to 0.
        far_depth_thresh (int, optional): Maximum considered depth. Defaults to None.
    Returns:
        np.ndarray: Depth quantized image.
    """

    mask = image > near_depth_thresh
    if far_depth_thresh != None:
        mask = np.logical_and(mask, image < far_depth_thresh)

    d_max = max(1, image.max())
    if np.any(mask):
        d_min = image[mask].min()
    else:
        d_min = 0

    g_stride = int((g_max - g_min) / eta)
    return np.where(mask, g_min + np.round(eta * (image - d_min) / (d_max - d_min + 1e-12)) * g_stride, 0).astype(np.uint8)
