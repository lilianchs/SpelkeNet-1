import cv2
import numpy as np
import torch


def get_dot_product_map(avg_flow, flow_cond_with_obj):
    # Compute flow direction vector
    dx, dy = np.array(flow_cond_with_obj[-1][2:]) - np.array(flow_cond_with_obj[-1][0:2])
    direction = np.array([dx, dy])

    # Compute dot product between avg_flow and direction vector
    dot_prod = torch.sum(
        avg_flow * torch.tensor(direction, dtype=avg_flow.dtype, device=avg_flow.device)[None, None, :], dim=-1)
    # dot_prod_np = #dot_prod.cpu().numpy()

    return dot_prod

def square_crop_with_padding(image, seg_mask, probe_point, padding=25, out_size=256):
    """
    Crop a square region around a binary segment in the image with padding.

    Args:
        image (H, W, C) or (C, H, W): Input image as numpy array or torch tensor
        seg_mask (H, W): Binary segmentation mask (numpy or torch)
        padding (int): Extra pixels to pad on all sides

    Returns:
        Cropped image and mask (both as same type as input)
    """

    x, y = probe_point

    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy() if image.ndim == 3 else image.cpu().numpy()
        seg_mask_np = seg_mask.cpu().numpy() if isinstance(seg_mask, torch.Tensor) else seg_mask
    else:
        image_np = image
        seg_mask_np = seg_mask

    H, W = image_np.shape[:2]

    y_indices, x_indices = np.where(seg_mask_np > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Segmentation mask is empty.")

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Compute center and half side
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    half_size = max(x_max - x_min, y_max - y_min) // 2 + padding

    # Initial square crop bounds
    x_start = center_x - half_size
    x_end = center_x + half_size + 1
    y_start = center_y - half_size
    y_end = center_y + half_size + 1

    # Shift box to stay within image bounds
    def adjust_bounds(start, end, max_val):
        length = end - start
        if start < 0:
            end = min(end - start, max_val)
            start = 0
        if end > max_val:
            start = max(0, start - (end - max_val))
            end = max_val
        return start, start + length

    x_start, x_end = adjust_bounds(x_start, x_end, W)
    y_start, y_end = adjust_bounds(y_start, y_end, H)

    # Ensure square crop (pad the smaller side)
    crop_width = x_end - x_start
    crop_height = y_end - y_start
    diff = abs(crop_height - crop_width)

    if crop_width < crop_height and x_end + diff <= W:
        x_end += diff
    elif crop_width < crop_height:
        x_start = max(0, x_start - diff)
    elif crop_height < crop_width and y_end + diff <= H:
        y_end += diff
    elif crop_height < crop_width:
        y_start = max(0, y_start - diff)

    cropped_img = image_np[y_start:y_end, x_start:x_end]
    cropped_mask = seg_mask_np[y_start:y_end, x_start:x_end]

    ratio = out_size / cropped_img.shape[0]

    cropped_img = cv2.resize(cropped_img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    x = (x - x_start) * ratio  # probe point in new 256x256 image
    y = (y - y_start) * ratio

    # Convert back to tensor if input was tensor
    if isinstance(image, torch.Tensor):
        cropped_img = torch.from_numpy(cropped_img).permute(2, 0, 1) if image.ndim == 3 else torch.from_numpy(
            cropped_img)
        cropped_mask = torch.from_numpy(cropped_mask)

    return cropped_img, cropped_mask, int(x), int(y), x_start, y_start, x_end, y_end, ratio



def convert_iterative_bboxes_to_absolute(bboxes, ratios):
    """
    Converts a chain of nested relative bboxes to absolute coordinates in the original image.

    Parameters:
    - bboxes: list of np.array([x_start, y_start, x_end, y_end]) relative to previous

    Returns:
    - List of np.array in absolute image coordinates
    """
    abs_bboxes = []
    offset = np.array([0, 0], dtype=np.float32)

    ratio = 1

    for ct, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.float32)
        if ct > 0:
            ratio *= 1 / ratios[ct - 1]
        # print(ratio)
        # Shift both start and end by the current offset (top-left of crop)
        abs_bbox = bbox.copy()
        abs_bbox = abs_bbox * ratio
        abs_bbox[0] += offset[0]  # x_start and x_end
        abs_bbox[2] += offset[0]
        abs_bbox[1] += offset[1]  # y_start and y_end
        abs_bbox[3] += offset[1]
        # print(abs_bbox)
        abs_bboxes.append(abs_bbox)

        # abs_bbox

        # Update offset to new crop's top-left
        offset = abs_bbox[:2]

    return [np.round(bbox).astype(np.int32) for bbox in abs_bboxes]


def resize_segment_to_original(segment, final_bbox_in_orig, original_image_shape):
    """
    Resizes and pastes a cropped segment map back into its position in the original image.

    Parameters:
    - segment: 2D numpy array (cropped segment map)
    - final_bbox_in_orig: (x_start, y_start, x_end, y_end) — bbox in original image coords
    - original_image_shape: (H, W) — shape of original image

    Returns:
    - full_segment_in_orig: 2D numpy array of shape (H, W) with segment placed appropriately
    """
    H, W = original_image_shape
    x_start, y_start, x_end, y_end = final_bbox_in_orig

    crop_width = min(x_end, W) - x_start
    crop_height = min(y_end, H) - y_start

    # print(crop_width, crop_height)

    # assert crop_width == crop_height, "Crop region must be square"

    # Resize the segment to match crop size in original image
    resized_segment = cv2.resize(segment, (crop_width, crop_height), interpolation=cv2.INTER_NEAREST)

    # Initialize full-size segment map

    full_segment = np.zeros((H, W), dtype=resized_segment.dtype)

    # print(y_start, y_end, x_start, x_end, resized_segment.shape, segment.shape, full_segment.shape)

    # Paste resized segment into the correct location
    full_segment[y_start:y_start + crop_height, x_start:x_start + crop_width] = resized_segment

    return full_segment


def get_random_coords_outside_segment(segment, N):
    """
    Returns N randomly sampled (x, y) coordinates outside the binary segment.
    `segment` should be a 2D numpy array of 0s and 1s, where 1 indicates the segment.
    """
    outside_mask = segment == 0
    y_coords, x_coords = np.where(outside_mask)
    coords = np.stack([x_coords, y_coords], axis=1)  # (N_all, 2)

    if len(coords) == 0:
        raise ValueError("No outside pixels found in the segment map.")

    N = min(N, len(coords))  # avoid index error
    sampled_indices = np.random.choice(len(coords), size=N, replace=False)
    sampled_coords = coords[sampled_indices]

    return sampled_coords


def sample_distant_point_on_segment(segment_map, point, min_dist=8, max_dist=20, max_tries=100):
    """
    Sample a point on the segment at least `min_dist` and at most `max_dist` pixels away.

    Args:
        segment_map (torch.Tensor): (H, W) binary tensor with segment as 1s.
        point (tuple): (y, x) coordinates.
        min_dist (int): minimum L2 distance from the point.
        max_dist (int): maximum L2 distance from the point.
        max_tries (int): number of tries before fallback.

    Returns:
        (y_new, x_new): sampled point on the segment or original point if none found.
    """
    H, W = segment_map.shape
    x0, y0 = point

    for _ in range(max_tries):
        angle = torch.rand(1).item() * 2 * np.pi
        radius = torch.empty(1).uniform_(min_dist, max_dist).item()
        dy = int(round(radius * np.sin(angle)))
        dx = int(round(radius * np.cos(angle)))

        y_new, x_new = y0 + dy, x0 + dx

        if 0 <= y_new < H and 0 <= x_new < W:
            if segment_map[y_new, x_new] == 1:
                return torch.tensor([x_new, y_new])

    return False
