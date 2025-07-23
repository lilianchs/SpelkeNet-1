import cv2
import math
import numpy as np
import torch
from PIL import ImageOps, Image
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
from matplotlib.patches import Polygon
from scipy.optimize import linear_sum_assignment
from skimage.measure import find_contours
from skimage.transform import resize
from spelke_net.utils.flow import flow_to_image

def batched_iou(x, y=None):
    """IoU between (B, H, W)"""
    if y is None:
        y = x

    xp = x[:, None]
    yp = y[None]

    intersection = (xp & yp).sum(axis=(-1, -2))
    union = (xp | yp).sum(axis=(-1, -2))

    return intersection / union


def evaluate_AP_AR_single_image(pred_segments, gt_segments):
    """
    Compute Average Precision (AP) and Average Recall (AR) for a single image.
    Precision and Recall are computed over IoU=.50:.05:.95.

    The procedure is as follows:
      1. Assign predicted segments to one of the gt segments, such that
        the IoU between gt and pred in each bin is maximized (globally).
      2. Compute True Positives, i.e. number of bins such that
        the IoU between gt and pred is greater than IoU threshold.
      3. Compute Precision and Recall, average across IoU thresholds.

    Arguments:
      pred_segments (N, H, W): N predicted segment masks.
      gt_segments (M, H, W): M ground truth segment masks.

    Returns:
      (dict(str: any)): A dictionary containing evaluation results.
    """

    iou_mat = batched_iou(gt_segments, pred_segments)
    gt_inds, pred_inds = linear_sum_assignment(1. - iou_mat)

    ious = iou_mat[gt_inds, pred_inds]

    num_gt_segments = gt_segments.shape[0]
    num_pred_segments = pred_segments.shape[0]

    precisions = []
    recalls = []

    thresholds = np.arange(start=0.50, stop=0.95, step=0.05)

    for i, iou_thresh in enumerate(thresholds):
        tp = np.count_nonzero(ious >= iou_thresh)

        if num_pred_segments == 0:
            precisions.append(0)
        else:
            precisions.append(tp / num_pred_segments)

        if num_gt_segments == 0:
            recalls.append(0)
        else:
            recalls.append(tp / num_gt_segments)

    return {
        'AP': np.mean(precisions),
        'AR': np.mean(recalls),
        'assignments': [gt_inds, pred_inds],
        'iou_mat': iou_mat,
        'thresholds': thresholds
    }


def sample_from_heatmap(heatmap):
    """
    Sample a (row, col) coordinate from a 2D heatmap.

    Args:
        heatmap (np.ndarray): 2D array of shape [H, W] with non-negative values.

    Returns:
        tuple: (row, col) index sampled from the heatmap.
    """
    # Flatten and normalize
    flat = heatmap.flatten()
    prob = flat / (np.sum(flat) + 1e-8)  # avoid division by zero

    # Sample from flattened index
    idx = np.random.choice(len(flat), p=prob)

    # Convert back to 2D index
    row, col = np.unravel_index(idx, heatmap.shape)
    return row, col



def resize_segments_np(segment_masks, size=256):
    """Resize segment masks the same way we would resize Images.
    It also removes segment masks that are cropped out."""
    segment_masks = (np.array(
        [np.array(ImageOps.fit(Image.fromarray(x * 255), (size, size))) for x in segment_masks]) / 255).astype('uint8')
    segment_masks = segment_masks[np.any(segment_masks, axis=(-1, -2))]

    return segment_masks



def plot_expected_displacement_map(ax, flow_map, mag_thresh=3, num_samples=18, arrow_color=(0, 0, 1)):
    """
    Overlay expected displacement arrows on a given axes.

    Args:
        ax: Matplotlib axis to plot on.
        flow_map: (H, W, 2) array of flow vectors.
        mag_thresh: Threshold to filter out low-magnitude flow.
        num_samples: Number of flow points to sample.
        arrow_color: Color of arrows (RGB tuple).
    Returns:
        ax: Updated matplotlib axis.
    """
    mag = np.linalg.norm(flow_map, axis=2)
    nonzero = np.argwhere(mag > mag_thresh)

    if len(nonzero) == 0:
        return ax

    sampled = nonzero[np.random.choice(len(nonzero), size=min(num_samples, len(nonzero)), replace=False)]

    for y, x in sampled:
        dx, dy = flow_map[y, x]
        ax.arrow(x, y, dx, dy, color=arrow_color, head_width=2, head_length=3)

    return ax

def plot_image_with_virtual_poke(ax, img, flow_cond_poke, dot_color='yellow', arrow_color='yellow'):
    """
    Overlay a virtual poke (point + vector) on the input image.
    """
    x1, y1, x2, y2 = flow_cond_poke
    ax.imshow(img)
    circle = plt.Circle((x1, y1), radius=5, facecolor=dot_color, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.arrow(x1, y1, x2 - x1, y2 - y1, color=arrow_color, head_width=3, head_length=5, linewidth=2)
    return ax


def offset_multiple_centroids(centroids, N, min_mag=10.0, max_mag=25.0):
    """
    Applies N offset vectors to each centroid in centroids, using same directions.

    centroids: Tensor of shape (M, 2) in (y, x) format
    Returns: Tensor of shape (N, M, 4) as [y1, x1, y2, x2]
    """
    device = centroids.device
    M = centroids.shape[0]

    # Angles and directions: shape (N,)
    angles = torch.arange(N, device=device) * (2 * math.pi / N)

    dx_unit = torch.cos(angles)  # (N,)
    dy_unit = torch.sin(angles)  # (N,)

    # Sample one magnitude per direction (shared across all centroids)
    magnitudes = torch.rand(N, device=device) * (max_mag - min_mag) + min_mag  # (N,)
    dx = magnitudes * dx_unit  # (N,)
    dy = magnitudes * dy_unit  # (N,)

    return dx, dy #


def find_nearest_flat_indices(logits: torch.Tensor, start: int, end: int):
    """
    Given logits of shape [N, C, T], find positions where logits[:, 0, start:end].sum(1) == 0,
    and return the nearest grid point's flattened index in a 32x32 grid (excluding the point itself).

    Returns:
        pts_flat: 1D tensor of original flat indices
        nearest_flat: 1D tensor of nearest grid point flat indices
    """
    # Find original points with zero logits
    pts_flat = torch.where(logits[:, 0, start:end].sum(1) == 0)[0]
    pts_x = pts_flat // 32
    pts_y = pts_flat % 32
    pts = torch.stack([pts_x, pts_y], dim=1)  # [M, 2]

    # Generate full 32x32 grid
    grid_x, grid_y = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(pts.device).float()  # [1024, 2]

    # Compute distances from each pt to all grid points
    dists = torch.cdist(pts.float(), grid_points, p=2)  # [M, 1024]

    # Exclude self-match
    for i in range(pts.shape[0]):
        self_idx = (grid_points == pts[i]).all(dim=1).nonzero(as_tuple=True)[0]
        dists[i, self_idx] = float('inf')

    # Find closest grid point
    min_indices = torch.argmin(dists, dim=1)
    nearest_pts = grid_points[min_indices].to(dtype=torch.int64)  # [M, 2]

    # Convert to flat indices
    nearest_flat = nearest_pts[:, 0] * 32 + nearest_pts[:, 1]

    return pts_flat, nearest_flat


def compute_avg_flow_from_logits(logits, start, end, token_to_flow_dict):
    """
    Computes average flow direction from logits using log-space softmax and a token-to-flow mapping.

    Args:
        logits (Tensor): Logits of shape (B, 1, V).
        start (int): Start index of flow range.
        end (int): End index of flow range.
        token_to_flow_dict (dict): Mapping from token ID to 2D flow vector.
        flow_to_image (function): Function to convert flow tensor to visual image.

    Returns:
        np.ndarray: Flow visualization image of shape (H, W, 3).
    """
    # Extract token IDs and flow vectors
    token_ids = np.array(list(token_to_flow_dict.keys()))
    flow_mapp = np.stack([token_to_flow_dict[x] for x in token_ids], axis=0)


    pts, nearest_pts = find_nearest_flat_indices(logits, start, end)

    logits[pts, 0] = logits[nearest_pts, 0]

    # Slice and gather logits
    logits = logits[:, 0]  # shape: (B, V)
    logits = logits[:, start:end]  # shape: (B, V_range)
    logits_valid = logits[:, token_ids]  # shape: (B, len(token_ids))

    # Log-space softmax and expected flow computation
    log_probs = logits_valid - torch.logsumexp(logits_valid, dim=1, keepdim=True)

    avg_flow = torch.einsum('bi,ijkl->bjkl', torch.exp(log_probs), torch.tensor(flow_mapp, dtype=torch.bfloat16))

    # avg_flow = torch.exp(log_probs) @ torch.tensor(flow_mapp, dtype=torch.bfloat16)

    # print(avg_flow.shape)  # .shape, flow_mapp.shape)

    avg_flow = avg_flow.view(32, 32, 4, 4, 2)
    avg_flow = avg_flow.permute(0, 2, 1, 3, 4).contiguous()
    avg_flow = avg_flow.view(128, 128, 2).to(torch.float32) / 2
    # avg_flow = avg_flow.permute(0, 2, 1, 3, 4).contiguous()
    # avg_flow = avg_flow.view(256, 256, 2).to(torch.float32)

    # Reshape and convert to image
    # avg_flow = avg_flow.view(32*4, 32*4, 2).to(torch.float32)
    flow_pred = flow_to_image(avg_flow.cpu().numpy())

    return avg_flow, flow_pred



def threshold_heatmap(heatmap):

    # Step 1: Min-Max Normalize to [0, 255]
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    heatmap_norm = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)  # Avoid divide-by-zero
    heatmap_scaled = (heatmap_norm * 255).astype(np.uint8)

    # Step 2: Apply Otsu thresholding
    _, thresh = cv2.threshold(heatmap_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh // 255


def plot_segment_overlay(ax, img, segment_mask, poke_xy=None, overlay_alpha=0.7):
    """
    Overlay a binary segment mask on an image with yellow transparent overlay, borders, and optional poke point.

    Args:
        ax: Matplotlib axis to plot on
        img: (H, W, 3) RGB image
        segment_mask: (H_seg, W_seg) binary mask, can be smaller than image
        poke_xy: Optional (x, y) tuple for poke point (in image coordinates)
        overlay_alpha: Alpha for the yellow overlay
    """

    # Resize mask to match image shape
    resized_mask = resize(segment_mask, (img.shape[0], img.shape[1]), order=0, preserve_range=True).astype(bool)

    # Create transparent-yellow colormap
    yellow_cmap = ListedColormap([[1, 1, 0, 0], [1, 1, 0, 1.0]])

    # Show image and overlay
    ax.imshow(img)
    ax.imshow(resized_mask, cmap=yellow_cmap, alpha=overlay_alpha)

    # Draw black border
    contours = find_contours(resized_mask.astype(float), 0.5)
    for contour in contours:
        polygon = Polygon(contour[:, [1, 0]], closed=True, edgecolor='black', facecolor='none', linewidth=2)
        ax.add_patch(polygon)

    # Draw poke point as red circle with black edge
    if poke_xy is not None:
        x, y = poke_xy
        poke_circle = Circle(
            (x, y), radius=4, facecolor='red',
            edgecolor='black', linewidth=2, zorder=10
        )
        ax.add_patch(poke_circle)

    ax.axis('off')
    # ax.set_title("Segment Overlay")
    return ax

