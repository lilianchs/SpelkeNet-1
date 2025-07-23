import torch
import numpy as np
import torch
import matplotlib.pyplot as plt

def unproject_pixels(pts, depth, intrinsics):
    '''
    pts: [N, 2] pixel coords
    depth: [N, ] depth values
    returns: [N, 3] world coords
    '''

    img_pixs = pts[:, [1, 0]].T
    img_pix_ones = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))

    img_inv = np.linalg.inv(intrinsics)
    cam_img_mat = np.dot(img_inv, img_pix_ones)
    # print(img_pix_ones)

    points_in_cam_ = np.multiply(cam_img_mat, depth.reshape(-1))

    return points_in_cam_.T


def project_pixels(pts, intrinsics):
    '''
    pts: [N, 2] pixel coords
    depth: [N, ] depth values
    returns: [N, 3] world coords
    '''

    img_pixs = pts.T  #

    img_inv = intrinsics[:3, :3]

    pts_in_cam = np.dot(img_inv, img_pixs)

    pts_in_cam = pts_in_cam // pts_in_cam[-1:, :]

    pts_in_cam = pts_in_cam[[1, 0, 2], :]

    return pts_in_cam.T


def compute_r_t(points_set1, points_set2):
    # Ensure inputs are numpy arrays
    points_set1 = np.array(points_set1)
    points_set2 = np.array(points_set2)

    # Compute centroids
    centroid1 = np.mean(points_set1, axis=0)
    centroid2 = np.mean(points_set2, axis=0)

    # Center the points
    centered1 = points_set1 - centroid1
    centered2 = points_set2 - centroid2

    # Compute covariance matrix
    H = np.dot(centered1.T, centered2)

    # Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure proper orientation (det(R) should be 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute translation
    T = centroid2 - np.dot(R, centroid1)

    return R, T


def get_true_pixel_coords(segment_map):
    # Get the indices of True values
    rows, cols = np.where(segment_map)
    # Combine into (row, col) coordinates
    coords = np.column_stack((rows, cols))
    return coords


def convert_segment_map_to_3d_coords(segment_map, depth_map, K):
    '''
    segment_map: [H, W]
    depth_map: [H, W]
    K: intrinsics
    returns: [N, 3] 3D coords
    '''

    # get 2D pixel coordinates from segment_map
    segment_coords_in_pixels_img0 = get_true_pixel_coords(segment_map)

    # get depth at those locations
    depth_coords = depth_map[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]]

    # unproject points
    segment_coords_in_3d_img0 = unproject_pixels(segment_coords_in_pixels_img0, depth_coords, K)

    return segment_coords_in_3d_img0, segment_coords_in_pixels_img0




def get_dense_flow_from_segment_depth_RT(segment_map, depth_map, R, T, K):
    '''
    segment_map: [H, W]
    depth_map: [H, W]
    R: [3, 3] rotation matrix
    T: [3, ] translation vector
    returns: dense flow map [H, W]
    '''

    # get 2D pixel coordinates from segment_map
    segment_coords_in_pixels_img0 = get_true_pixel_coords(segment_map)

    # get depth at those locations
    depth_coords = depth_map[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]]

    # unproject points
    segment_coords_in_3d_img0 = unproject_pixels(segment_coords_in_pixels_img0, depth_coords, K)

    # transform with RT
    segment_coords_in_3d_img1 = np.dot(R, segment_coords_in_3d_img0.T) + T[:, None]
    segment_coords_in_3d_img1 = segment_coords_in_3d_img1.T

    # project onto image
    segment_coords_in_pixels_img1 = project_pixels(segment_coords_in_3d_img1, K)

    # compute flow vectors
    flow_ = segment_coords_in_pixels_img1[:, :2] - segment_coords_in_pixels_img0

    # make flow map
    flow_map = np.zeros([segment_map.shape[0], segment_map.shape[1], 2])
    flow_map[segment_coords_in_pixels_img0[:, 0], segment_coords_in_pixels_img0[:, 1]] = flow_

    #make fake segment map
    segment_coords_in_pixels_img1 = segment_coords_in_pixels_img1.astype(int)
    #clip to image size
    segment_coords_in_pixels_img1 = np.clip(segment_coords_in_pixels_img1, 0, segment_map.shape[0] - 1)
    segment_map = np.zeros([segment_map.shape[0], segment_map.shape[1]])
    segment_map[segment_coords_in_pixels_img1[:, 0], segment_coords_in_pixels_img1[:, 1]] = 1

    return flow_map,  segment_coords_in_pixels_img0, segment_coords_in_3d_img1, segment_map

import cv2
def combine_dilated_bounding_boxes(seg1, seg2, kernel_size=5, iterations=1):
    """
    Combines two binary segmentation maps by performing the following steps:
    1. Dilates each segmentation map.
    2. Finds the bounding box for the dilated region.
    3. Creates a new binary map for each bounding box.
    4. Adds the two bounding box maps to produce a combined map.

    Parameters:
        seg1 (np.ndarray): First binary segmentation map.
        seg2 (np.ndarray): Second binary segmentation map.
        kernel_size (int): Size of the square kernel used for dilation.
        iterations (int): Number of dilation iterations.

    Returns:
        combined_map (np.ndarray): Combined binary map with bounding boxes.
    """
    # Create the structuring element for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilate both segmentation maps
    dilated1 = cv2.dilate(seg1, kernel, iterations=iterations)
    dilated2 = cv2.dilate(seg2, kernel, iterations=iterations)

    # Nested helper function to extract the bounding box from a binary image
    def get_bounding_box(binary_img):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        return int(x_min), int(y_min), int(x_max), int(y_max)

    # Get bounding boxes for each dilated map
    bbox1 = get_bounding_box(dilated1)
    bbox2 = get_bounding_box(dilated2)

    # Initialize bounding box maps with zeros
    bbox_map1 = np.zeros_like(seg1, dtype=np.uint8)
    bbox_map2 = np.zeros_like(seg2, dtype=np.uint8)

    if bbox1 is not None:
        x_min, y_min, x_max, y_max = bbox1
        bbox_map1[y_min:y_max, x_min:x_max] = 1

    if bbox2 is not None:
        x_min, y_min, x_max, y_max = bbox2
        bbox_map2[y_min:y_max, x_min:x_max] = 1

    # Combine the bounding box maps and ensure the result remains binary (0 or 1)
    combined_map = np.clip(bbox_map1 + bbox_map2, 0, 1)
    return combined_map





def to_tensor_flow(flow):
    '''
    flow: [H, W, 2]
    returns torch tensor: [1, 2, H, W]
    '''

    flow = torch.from_numpy(flow).permute(2, 0, 1)[None]

    return flow


def to_tensor_segment(seg):
    '''
    flow: [H, W]
    returns torch tensor: [1, 1, H, W]
    '''

    flow = torch.from_numpy(seg)[None, None]

    return flow


def downsample_flow(flow_tensor_cuda, kernel_size=4, stride=4):
    '''
    :param flow_tensor_cuda: [1, 2, 256, 256]
    :return:
    '''
    magnitude = torch.sqrt(flow_tensor_cuda[0, 0] ** 2 + flow_tensor_cuda[0, 1] ** 2)

    # max pool flow with 4x4 kernel
    magnitude, indices = torch.nn.functional.max_pool2d(magnitude[None, None], kernel_size=kernel_size,
                                                        stride=kernel_size,
                                                        return_indices=True)

    # Use the saved indices to pool the additional_tensor
    flowmap_x = flow_tensor_cuda[0, 0].view(-1).gather(0, indices.view(-1)).view(magnitude.shape)
    flowmap_y = flow_tensor_cuda[0, 1].view(-1).gather(0, indices.view(-1)).view(magnitude.shape)

    flow_tensor_cuda_maxpooled = torch.concatenate([flowmap_x, flowmap_y], dim=1)  # [0]

    return flow_tensor_cuda_maxpooled, indices[0, 0]


def get_flattened_index_from_2d_index(indices, size):
    '''
    indices: [N, 2]
    '''

    return indices[:, 0] * size + indices[:, 1]


def downsample_and_scale_flow(flow_map, to_tensor=True):
    '''
    flow_map: [H, W, 2]
    '''
    if to_tensor:
        flow_map_orig = to_tensor_flow(flow_map)
    else:
        flow_map_orig = flow_map

    flow_map_, indices = downsample_flow(flow_map_orig.cuda(), 4, 4)

    flow_map, indices = downsample_flow(flow_map_.cuda(), 4, 4)

    flow_map = flow_map / (flow_map_orig.shape[-1] / 256)

    #clip flow
    flow_map = torch.clamp(flow_map, -256, 256)

    return flow_map, indices



def get_unmask_indices_from_flow_map(flow_map, num_fg_flows=80, num_bg_flows=20, use_mag=False, new_sampling_method=False):
    '''
    flow_map: [1, 2, H, W]
    num_fg_flows
    num_bg_flows
    '''

    flow_map = flow_map[0].permute(1, 2, 0).detach().cpu().numpy()

    if use_mag:
        flow_map_mag = np.sqrt(flow_map[:, :, 0] ** 2 + flow_map[:, :, 1] ** 2)
        segment_map = flow_map_mag > 1
    else:
        # get segment map from flow
        segment_map = np.abs(flow_map).mean(-1) > 0
    segment_map_tensor = to_tensor_segment(segment_map)

    # downsample segment map to 32 x 32 resolution

    segment_map_tensor_fg = torch.nn.functional.max_pool2d(segment_map_tensor.float(), kernel_size=2, stride=2)[0, 0]
    segment_map_tensor_fg = segment_map_tensor_fg.bool().cpu().numpy()
    segment_map_tensor_bg = ~segment_map_tensor_fg

    if new_sampling_method:
        segment_map_tensor_fg = -torch.nn.functional.max_pool2d(-segment_map_tensor.float(), kernel_size=2, stride=2)[0, 0]
        segment_map_tensor_fg = segment_map_tensor_fg.bool().cpu().numpy()

        segment_map_tensor_bg = torch.nn.functional.max_pool2d(segment_map_tensor.float(), kernel_size=2, stride=2)[0, 0]
        segment_map_tensor_bg = ~segment_map_tensor_bg.bool().cpu().numpy()


    inds_fg = get_flattened_index_from_2d_index(get_true_pixel_coords(segment_map_tensor_fg), segment_map_tensor_fg.shape[0])

    inds_bg = get_flattened_index_from_2d_index(get_true_pixel_coords(segment_map_tensor_bg), segment_map_tensor_fg.shape[0])

    # randomly select part of these inds
    indices_fg = np.arange(len(inds_fg))
    np.random.shuffle(indices_fg)
    indices_fg = indices_fg[:num_fg_flows]

    indices_bg = np.arange(len(inds_bg))
    np.random.shuffle(indices_bg)
    indices_bg = indices_bg[:num_bg_flows]

    inds_fg = inds_fg[indices_fg]
    inds_bg = inds_bg[indices_bg]

    indices = np.concatenate([inds_fg, inds_bg])

    return list(indices)




def plot_flow_visualizations(image0_downsampled,  rgb1_pred,  output_path="counterfactual_image.png"):
    """
    Function to plot the flow visualizations.
    Parameters:
    - image0_downsampled: Downsampled version of the first image.
    - rgb1_pred: Counterfactual RGB prediction from the model.
    - output_path: Path to save the output plot.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Sparse flow visualization
    ax[0].imshow(image0_downsampled)
    ax[0].set_title("Input Image")
    ax[0].axis('off')

    # Plot the counterfactual RGB prediction
    ax[1].imshow(rgb1_pred)
    ax[1].set_title("Edited image")
    ax[1].axis('off')

    # Save the figure
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


