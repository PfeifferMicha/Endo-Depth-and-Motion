from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


def depth_to_color(
    depth_np: np.ndarray,
    cmap: str = "gist_rainbow",
    max_depth: Optional[float] = None,
    min_depth: Optional[float] = None,
) -> np.ndarray:
    """Converts a depth image to color using the specified color map.

    Arguments:
        depth_np {np.ndarray} -- Depth image/Or inverse depth image. [HxW]
    Keyword Arguments:
        cmap {str} -- Color map to be used. (default: {"gist_rainbow"})

    Returns:
        np.ndarray -- Color image [HxWx3]
    """
    # -- Set default arguments
    depth_np[np.isinf(depth_np)] = np.nan
    if max_depth is None:
        max_depth = np.nanmax(depth_np)
    if min_depth is None:
        min_depth = min_depth or np.nanmin(depth_np)

    cm = plt.get_cmap(cmap, lut=1000)
    depth_np_norm = (depth_np - min_depth) / (max_depth - min_depth)
    colored_depth = cm(depth_np_norm)

    return (colored_depth[:, :, :3] * 255).astype(np.uint8)


def depth_edge_mask_from_angles(depth, camera_parameters, angle_threshold=10):
    xyz = depth_to_xyz(depth, camera_parameters)
    edge_mask = find_parallel_lines_fast(xyz, angle_threshold)
    return edge_mask

def depth_to_xyz(depth, camera_parameters):
    fx = camera_parameters['fx']
    fy = camera_parameters['fy']
    cx = camera_parameters['cx']
    cy = camera_parameters['cy']

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = depth
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy

    xyz = np.stack((x, y, z), axis=-1)
    return xyz

def find_parallel_lines_fast(matrix, angle_threshold=10):
    # matrix is a (w, h, 3) matrix of the xyz positions of the pixels in the depth image

    shifted_x_0 = np.roll(matrix, shift=-1, axis=1)
    shifted_x_1 = np.roll(matrix, shift=1, axis=1)
    shifted_y_0 = np.roll(matrix, shift=-1, axis=0)
    shifted_y_1 = np.roll(matrix, shift=1, axis=0)

    # Calculate the difference along the x and y axes
    line_x_0 = shifted_x_0 - matrix
    line_x_1 = shifted_x_1 - matrix
    line_y_0 = shifted_y_0 - matrix
    line_y_1 = shifted_y_1 - matrix

    # Calculate the angles between the lines and the camera vector for both x and y directions
    angle_x_0 = angle_between_lines(matrix, line_x_0)
    angle_x_1 = angle_between_lines(matrix, line_x_1)
    angle_y_0 = angle_between_lines(matrix, line_y_0)
    angle_y_1 = angle_between_lines(matrix, line_y_1)

    # Create the parallel array by checking if the angles are less than or equal to the angle_threshold
    parallel = (angle_x_0 <= angle_threshold)|(angle_x_1 <= angle_threshold)|(angle_y_0 <= angle_threshold)|(angle_y_1 <= angle_threshold)

    return parallel

def angle_between_lines(camera_vector, vector_line):
    # Calculate the dot product between the two vectors
    dot_product = np.sum(vector_line * camera_vector, axis=-1)

    # Calculate the magnitudes (norms) of the vectors
    norm_line = np.linalg.norm(vector_line, axis=-1)
    norm_camera = np.linalg.norm(camera_vector, axis=-1)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_line * norm_camera)

    # Ensure the value is within the valid range for arccos due to floating-point errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in radians
    theta_radians = np.arccos(cos_theta)

    # Convert the angle to degrees
    theta_degrees = np.degrees(theta_radians)

    return theta_degrees

