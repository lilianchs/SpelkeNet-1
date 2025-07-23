"""
Utility function for converting data into seqeunce of tokens and positional indexes.
"""
from typing import Tuple

import numpy as np
import torch


def add_patch_indexes(patches: torch.LongTensor, start_idx: int) -> torch.LongTensor:
    """
    Adds patch indexes to the ordered tensor of patches provided starting at the given index.

    Parameters:
        patches (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis
        start_idx (int): Starting index for the patch indexes.
    
    Returns:
        patches_with_indexes (torch.LongTensor) of shape B, N, P+1:
            Tensor of N patches each of size P with an additional index at the end
    """
    indexes = torch.arange(start_idx, start_idx + patches.shape[1]).reshape(1, -1, 1).to(patches.device)
    patches_with_indexes = torch.cat([indexes, patches], axis=2)
    return patches_with_indexes


def get_pos_idxs(tokens: torch.LongTensor, start_idx: int) -> torch.LongTensor:
    """
    Generate positional indexes for the sequence of tokens.

    Parameters:
        tokens (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis
        start_idx (int): Starting index for the positional indexes.
    
    Returns:
        pos_idx (torch.LongTensor) of shape B, N, P:
            Tensor of positional indexes for the sequence of tokens
    """
    # Create positional indexes for the sequence of tokens
    pos_idx = torch.arange(start_idx, start_idx + tokens.numel()).reshape(tokens.shape).to(tokens.device)
    return pos_idx


def shuffle_and_trim_values_and_positions(
        tokens: torch.LongTensor, positions: torch.LongTensor, 
        mask: float = 0.0, shuffle: bool = True, shuffle_order: np.ndarray = None
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Shuffle the tokens and positions along the N (1st) axis and remove mask amount of the tokens.

    Parameters:
        tokens (torch.LongTensor) of shape B, N, P:
            Tensor of N patches each of size P in sorted order along the N axis
        positions (torch.LongTensor) of shape B, N, P:
            Tensor of positional indexes for the sequence of tokens
        mask (float): Amount of the tokens to remove
        shuffle (bool): Whether to shuffle the tokens or not

    Returns:
        shuffled_tokens (torch.LongTensor) of shape B, N, P:
            Shuffled tensor of N patches each of size P in sorted order along the N axis
        shuffled_positions (torch.LongTensor) of shape B, N, P:
            Shuffled tensor of positional indexes for the sequence of tokens
    """
    # Shuffle patches on the 1st axis, as well as positions, if shuffle is True
    if shuffle_order is None:
        if shuffle:
            shuffle_order = np.random.permutation(tokens.shape[1])
        else:
            shuffle_order = np.arange(tokens.shape[1])
    else:
        assert len(shuffle_order) == tokens.shape[1], \
            (f"Provided shuffle order does not account for all tokens in sequence. "
             f"Expected {tokens.shape[1]} indices, got {len(shuffle_order)}.")

    # Remove mask amount of the patches
    shuffle_order = shuffle_order[:int(shuffle_order.shape[0] * (1-mask))]
    shuffled_tokens = tokens[:, shuffle_order]
    shuffled_positions = positions[:, shuffle_order]
    return shuffled_tokens, shuffled_positions
