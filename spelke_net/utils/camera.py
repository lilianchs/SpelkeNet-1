"""
Utilities for performing 3D transformations and camera motion.
"""

import torch
import numpy as np
import spelke_net.utils.transforms as py3d_t


def get_camera_orientation_dict_from_threepoints_depth_intrinsics(threepoints, intrinsics, depthmap):
    threepoints_tensor = torch.tensor(threepoints, dtype=torch.float32).unsqueeze(-1)
    threepoints_homogeneous = torch.cat((threepoints_tensor, torch.ones_like(threepoints_tensor[:, [0], :])), dim=1)
    intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
    threepoints_in_camera_unnormalized = torch.einsum('bij,bjk->bik', torch.inverse(intrinsics_tensor), threepoints_homogeneous)
    # breakpoint()
    # threepoints_in_camera_normalized = threepoints_in_camera_unnormalized / torch.norm(threepoints_in_camera_unnormalized, dim=1, keepdim=True)

    depth_of_threepoints = torch.tensor(depthmap[threepoints[:, 1], threepoints[:, 0]])
    threepoints_in_camera = threepoints_in_camera_unnormalized * depth_of_threepoints.unsqueeze(-1).unsqueeze(-1) # 3, 3, 1

    p1 = threepoints_in_camera[0].squeeze()
    p2 = threepoints_in_camera[1].squeeze()
    p3 = threepoints_in_camera[2].squeeze()

    v1 = p1 - p2 
    v2 = p3 - p2

    gravity_up = np.cross(v2, v1) # we are assuming the points are in counterclockwise order

    gravity_up_in_camera = gravity_up / np.linalg.norm(gravity_up)
    gravity_up_in_camera = torch.tensor(gravity_up_in_camera, dtype=torch.float32)
    camera_front = torch.tensor([0, 0, 1], dtype=torch.float32)
    world_x_in_camera = torch.cross(camera_front, gravity_up_in_camera)
    world_x_in_camera = world_x_in_camera / torch.norm(world_x_in_camera)
    world_y_in_camera = torch.cross(gravity_up_in_camera, world_x_in_camera)
    world_y_in_camera = world_y_in_camera / torch.norm(world_y_in_camera)


    transform_camera_from_world = torch.eye(4, dtype=torch.float32)
    transform_camera_from_world[:3, 2] = gravity_up_in_camera
    transform_camera_from_world[:3, 0] = world_x_in_camera
    transform_camera_from_world[:3, 1] = world_y_in_camera
    transform_world_from_camera = torch.inverse(transform_camera_from_world).unsqueeze(0)

    return {"transform_world_from_camera": transform_world_from_camera, 'transform_camera_from_world': transform_camera_from_world}

def pose_list_to_matrix(pose_list):
    """
    Convert a list of 6 floats [x_rot, y_rot, z_rot, x_trans, y_trans, z_trans]
    into a 4x4 transformation matrix (torch float tensor).
    """
    # Unpack the pose list
    x_rot, y_rot, z_rot, x_trans, y_trans, z_trans = pose_list

    # Convert rotations to torch tensors (assuming rotations are in radians)
    x_rot = torch.tensor(x_rot, dtype=torch.float32)
    y_rot = torch.tensor(y_rot, dtype=torch.float32)
    z_rot = torch.tensor(z_rot, dtype=torch.float32)

    # Compute cosine and sine of rotation angles
    cos_x, sin_x = torch.cos(x_rot), torch.sin(x_rot)
    cos_y, sin_y = torch.cos(y_rot), torch.sin(y_rot)
    cos_z, sin_z = torch.cos(z_rot), torch.sin(z_rot)

    # Rotation matrix around the x-axis
    Rx = torch.tensor([
        [1,     0,      0     ],
        [0,  cos_x, -sin_x],
        [0,  sin_x,  cos_x]
    ], dtype=torch.float32)

    # Rotation matrix around the y-axis
    Ry = torch.tensor([
        [ cos_y, 0, sin_y],
        [     0, 1,     0],
        [-sin_y, 0, cos_y]
    ], dtype=torch.float32)

    # Rotation matrix around the z-axis
    Rz = torch.tensor([
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [    0,      0, 1]
    ], dtype=torch.float32)

    # Combine the rotation matrices (order matters: R = Rz * Ry * Rx)
    R = Rz @ Ry @ Rx  # Matrix multiplication

    # Translation vector
    t = torch.tensor([x_trans, y_trans, z_trans], dtype=torch.float32)

    # Construct the 4x4 transformation matrix
    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def quantize_6dof_campose(
        campose: np.array, 
        translation_quantization: float = 0.001,
        translation_offset: np.array = 1001,
        rotation_quantization: float = 0.1,
        rotation_offset: np.array = 1801,
        translation_scale_quantization: float = 0.001,
        translation_scale_offset: float = 1,
        rotation_indexes: np.array = [0, 1, 2],
        translation_indexes: np.array = [3, 4, 5],
        translation_scale_index: int = 6,
        **kwargs
    ) -> np.array:
    """
    Quantize a 6-DOF camera pose vector

    Parameters:
        - campose: 6-DOF camera pose vector
        - translation_quantization: Quantization value for the translation
        - translation_offset: Offset value for the translation
        - rotation_quantization: Quantization value for the rotation
        - rotation_offset: Offset value for the rotation

    Returns:
        - quantized_campose: Quantized 6-DOF camera pose vector
    """

    # Extract the translation and rotation components of the camera pose
    translation = campose[translation_indexes]
    rotation = campose[rotation_indexes]
    if translation_scale_index is not None:
        translation_scale = campose[translation_scale_index]

    # Quantize the translation and rotation components
    quantized_translation = np.round(translation / translation_quantization)
    quantized_rotation = np.round(rotation / rotation_quantization)
    if translation_scale_index is not None:
        quantized_translation_scale = np.round(translation_scale / translation_scale_quantization)

    # Apply the offset values
    quantized_translation += translation_offset
    quantized_rotation += rotation_offset
    if translation_scale_index is not None:
        quantized_translation_scale += translation_scale_offset

    # Combine the quantized translation and rotation components into a single 6-DOF vector
    # quantized_campose = np.concatenate((quantized_translation, quantized_rotation))
    quantized_campose = np.zeros(6, dtype=np.int16) if translation_scale_index is None else np.zeros(7, dtype=np.int16)
    quantized_campose[translation_indexes] = quantized_translation
    quantized_campose[rotation_indexes] = quantized_rotation
    if translation_scale_index is not None:
        quantized_campose[translation_scale_index] = quantized_translation_scale

    # Convert the quantized camera pose to integers
    quantized_campose = quantized_campose.astype(np.int16)

    return quantized_campose


def transform_matrix_to_six_dof_axis_angle(matrix: np.ndarray, scale: bool = False) -> np.ndarray:
    """
    Convert a 4x4 transformation matrix to a 6-DOF vector (translation + axis-angle)

    Parameters:
        - matrix: 4x4 transformation matrix

    Returns:
        - six_dof_vector: 6-DOF vector (translation + axis-angle)
    """

    # Ensure the matrix is a numpy array
    matrix = np.array(matrix)
    
    # Extract the translation vector (last column of the first three rows)
    translation = matrix[:3, 3]
    
    # Extract the rotation matrix (first three rows and columns)
    rotation_matrix = matrix[:3, :3]

    # Convert to torch tensor with torch.float64
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float64)[None, ...]
    
    # Convert rotation matrix to Euler angles (XYZ sequence)
    rotation_so3 = py3d_t.so3_log_map(rotation_matrix)

    # Convert to numpy float32 and degrees, and remove the batch dimension
    rotation_so3 = rotation_so3[0].detach().numpy().astype(np.float32) * 180 / np.pi

    if scale:
        translation_scale = np.linalg.norm(translation, keepdims=True)
        translation_dir = translation / (translation_scale + 1e-6)
        six_dof_vector = np.concatenate((rotation_so3, translation_dir, translation_scale))
    else:
        # Combine translation and Euler angles into a single 6D vector
        six_dof_vector = np.concatenate((rotation_so3, translation))
        
    return six_dof_vector
