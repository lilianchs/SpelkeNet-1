import torchvision
import cv2
from PIL import Image, ImageOps
import torch
from einops import rearrange
import numpy as np
try:
    from moviepy.editor import ImageSequenceClip
except ImportError as e:
    from moviepy import ImageSequenceClip


def convert_from_16bit_color(image_16bit):
    """
    Converts a 16-bit color image back to a 24-bit RGB image.
    
    Args:
    image_16bit (numpy array): Input image in 16-bit color format.

    Returns:
    numpy array: Converted image in 24-bit RGB format.
    """
    # Extract the 5-bit Red, 6-bit Green, and 5-bit Blue channels
    r = (image_16bit >> 11) & 0x1F
    g = (image_16bit >> 5) & 0x3F
    b = image_16bit & 0x1F
    
    # Convert the channels back to 8-bit by left-shifting and scaling
    r = (r << 3) | (r >> 2)
    g = (g << 2) | (g >> 4)
    b = (b << 3) | (b >> 2)
    
    # Combine the channels into a 24-bit RGB image
    image_rgb = np.stack((b, g, r), axis=-1)
    
    return image_rgb.astype('uint8')


def patchify(imgs: torch.Tensor, patch_size: int = 4) -> torch.Tensor:
    """
    Convert images with no channel dimension into patches.

    Parameters:
        - imgs: Tensor of shape (B, H, W) 
            where B is the batch size, H is the height, and W is the width.
        - patch_size: 
            The size of each patch.

    Returns:
        Tensor of shape (B, L, patch_size**2)
            where L is the number of patches (H//patch_size * W//patch_size).
    """
    assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % patch_size == 0, \
        "Image dimensions must be square and divisible by the patch size."

    h = w = imgs.shape[1] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], h, patch_size, w, patch_size))
    x = torch.einsum('bhpwq->bhwpq', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2))
    return x



def unpatchify(patches):
    """
    Reconstruct images without color channel from patches.

    Parameters:
        - patches: Tensor of shape (B, L, patch_size**2)
            where B is the batch size, L is the number of patches, and patch_size**2 is the size of each patch.

    Returns:
        Tensor of shape (B, H, W) where H is the height and W is the width of the reconstructed image.
    """
    N_patches = int(np.sqrt(patches.shape[1]))  # number of patches along each dimension
    patch_size = int(np.sqrt(patches.shape[2]))  # patch size along each dimension

    rec_imgs = rearrange(patches, 'b (h w) (p0 p1) -> b h w p0 p1', h=N_patches, w=N_patches, p0=patch_size, p1=patch_size)
    rec_imgs = rearrange(rec_imgs, 'b h w p0 p1 -> b (h p0) (w p1)')
    
    return rec_imgs


def unpatchify_logits(patches):
    """
    Reconstruct images without color channel from patches.

    Parameters:
        - patches: Tensor of shape (B, L, patch_size**2, D)
            where B is the batch size, L is the number of patches, and patch_size**2 is the size of each patch.

    Returns:
        Tensor of shape (B, H, W, D) where H is the height and W is the width of the reconstructed image.
    """
    N_patches = int(np.sqrt(patches.shape[1]))  # number of patches along each dimension
    patch_size = int(np.sqrt(patches.shape[2]))  # patch size along each dimension

    rec_imgs = rearrange(patches, 'b (h w) (p0 p1) d -> b h w p0 p1 d', h=N_patches, w=N_patches, p0=patch_size,
                         p1=patch_size)
    rec_imgs = rearrange(rec_imgs, 'b h w p0 p1 d -> b (h p0) (w p1) d')

    return rec_imgs
