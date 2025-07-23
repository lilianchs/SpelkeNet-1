"""
Utility function for processing optical flow data
"""

from typing import Tuple

import numpy as np
import torch


def compute_quantize_flow(
        flow_values: torch.FloatTensor,
        input_size: int = 256,
        num_bins: int = 512,
    ) -> torch.LongTensor:
    """
    Quantizes the continuous optical flow values into discrete bins.

    Parameters:
    flow_values (FloatTensor): The original optical flow values to be quantized, [B, 2, H, W].
    input_size (int): The input size of the flow map (used to set the quantization range).
    num_bins (int): The number of bins for quantization (i.e., how many discrete levels the flow will be quantized into).

    Returns:
    LongTensor: The quantized flow values, where each value is an integer representing the bin index, [B, 2, H, W].
    """
    # Setting the flow range based on input_size, defining min_flow and max_flow.
    max_range = input_size
    min_flow, max_flow = -max_range, max_range

    # Normalize the flow values to the range [0, 1]
    normalized_flow = (flow_values - min_flow) / (max_flow - min_flow)

    # Ensure the normalized flow stays within the valid range [0, 1]
    normalized_flow = torch.clamp(normalized_flow, 0.0, 1.0)

    # Scale the normalized values to the number of bins
    scaled_flow = normalized_flow * (num_bins - 1)

    # Round to the nearest bin index and convert to long tensor
    return torch.round(scaled_flow).long()


def decode_flow_code(
        quantized_flow: torch.LongTensor,
        input_size: int = 256,
        num_bins: int = 512
    ) -> torch.FloatTensor:
    """
    Decodes the quantized optical flow values back into their original flow range.

    Parameters:
    quantized_flow (LongTensor): The quantized flow values to be decoded, [B, 2, H, W]
    input_size (int): The maximum expected range of the flow values (used to set the decoding range).
    num_bins (int): The number of bins used in quantization (i.e., how many discrete levels the flow was quantized into).

    Returns:
    flow_values (FloatTensor): The decoded flow values, scaled back to the original flow range. [B, 2, H, W]
    """
    # Setting the flow range based on input_size, defining min_flow and max_flow.
    max_range = input_size
    min_flow, max_flow = -max_range, max_range

    # Normalize the quantized values to the range [0, 1]
    normalized_flow = quantized_flow.float() / (num_bins - 1)

    # Scale the normalized values back to the range [min_flow, max_flow]
    flow_values = normalized_flow * (max_flow - min_flow) + min_flow

    return flow_values



def sample_flow_values_and_positions(
        tokens: torch.LongTensor,
        positions: torch.LongTensor,
        flows: torch.FloatTensor,
        num_flow_patches: int = 0.0,
        alpha: float = 0.75,
        exclude_mask: torch.BoolTensor = None
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Sample the tokens and positions along the N (1st) axis based on motion flow magnitudes, mask value, and alpha value.
    Only samples from positions where exclude_mask is False.

    Parameters:
        tokens (torch.LongTensor): Tensor containing N patches of size P shape [B, N, P]
        positions (torch.LongTensor): Tensor of positional indexes for the sequence of tokens,
                                      shape [B, N, P], with same dimensions as `tokens`.
        flows (torch.FloatTensor): Tensor containing the optical flow values corresponding to
                                   the tokens, shape [2, H, W].
        num_flow_patches (int): The number of flow patches to be sampled
        alpha (float): Proportion of patches to be selected based on their motion (optical flow).
                       Must be between 0 and 1 (0 means no motion-based patches, 1 means only
                       motion-based patches).
        exclude_mask (torch.BoolTensor): Boolean mask indicating which positions to exclude from sampling.
                                        True values will be excluded. Shape should match flow_magnitude.

    Returns:
        shuffled_tokens (torch.LongTensor): Shuffled tensor of N_m + N_r patches each of size P in
                                            the same order, shape [B, N_m + N_r, P].
        shuffled_positions (torch.LongTensor): Shuffled tensor of positional indexes for the
                                               sequence of tokens, shape [B, N_m + N_r, P].
    """

    # Ensure that alpha is within a valid range [0, 1]
    assert alpha >= 0. and alpha <= 1., "alpha should be between 0 and 1"

    # Compute the magnitude of the flow along the 0th dimension (e.g. for motion-based sorting)
    flow_magnitude = flows.norm(p=2, dim=0)  # [H, W]

    # Apply exclude mask if provided
    if exclude_mask is not None:
        flow_magnitude = flow_magnitude.clone()
        flow_magnitude[exclude_mask] = 0.0

    # Get valid indices where exclude_mask is False
    valid_indices = None if exclude_mask is None else (~exclude_mask).nonzero().squeeze()

    # Calculate the number of tokens to be selected based on motion (flows) and randomly
    num_motion = int(num_flow_patches * alpha)  # Number of motion-based tokens
    num_random = num_flow_patches - num_motion  # Number of randomly selected tokens

    # Sample tokens based on flow magnitude for motion-based selection
    if num_motion == 0:
        if valid_indices is not None:
            shuffle_order = valid_indices[torch.randperm(len(valid_indices))[:num_random]]
        else:
            all_indices = np.arange(tokens.shape[1])
            shuffle_order = np.random.permutation(all_indices)[:num_random]
    else:
        # Sample based on flow magnitude, but only from valid positions
        motion_order = torch.multinomial(flow_magnitude.flatten(), num_motion).numpy()  # [N_m]

        # Get remaining valid indices for random selection
        if valid_indices is not None:
            remaining_indices = np.setdiff1d(valid_indices.cpu().numpy(), motion_order)
        else:
            remaining_indices = np.setdiff1d(np.arange(tokens.shape[1]), motion_order)

        random_order = np.random.permutation(remaining_indices)[:num_random]
        shuffle_order = np.concatenate([motion_order, random_order], axis=0)  # [N_m + N_r]

    # Shuffle the tokens and positions based on the shuffle_order
    shuffled_tokens = tokens[:, shuffle_order]  # shape [B, N_m + N_r, P]
    shuffled_positions = positions[:, shuffle_order]  # shape [B, N_m + N_r, P]

    # Return the shuffled tokens and their corresponding positions
    return shuffled_tokens, shuffled_positions


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

