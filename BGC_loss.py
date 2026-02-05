import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

#####################################################
# MEMO #############################################
#####################################################

"""
Boundary Gradient Consistency Loss

Paper : Improvement of a Segmentation Network for Character Stroke Extraction 
        from Metal Movable Type Printed Documents
DOI   : https://doi.org/10.5573/ieie.2023.60.12.31

This loss function is designed to encourage smooth and consistent boundaries of the foreground
 (especially text regions) in segmentation tasks.
Since it only takes logits (predictions) as input, it should be used together with
 other loss functions during training.

In our experiments, the best performance was achieved when this loss was combined with
 Dice coefficient loss on old printed book datasets scanned at 600 dpi.

When the input resolution is too high, this loss tends to over-focus on details.
To mitigate this, max pooling is applied before computing the boundary gradients.
Therefore, the pooling scale should be adjusted according to the resolution of the input images.

Although average pooling or Gaussian blur can also be used, additional preprocessing steps
 (e.g., quantization or contour extraction) are required to compute boundary gradients properly.
"""

#####################################################
# Loss Functions ####################################
#####################################################

def BoundaryGradientConsistency_loss(logits, channel_dim: int=1, max_pool_scale: int =2):

    """
    Input: Prediction [Batch, Class, Height, Width]
    This version is designed for binary segmentation.
    """
    
    logits = softargmax(logits, dim=channel_dim)
    
    if max_pool_scale < 1:
        raise ValueError("ERROR: BDC_loss - max_pool_scale < 0")
    elif max_pool_scale > 1:
        logits = F.max_pool2d(logits.float(), kernel_size=max_pool_scale, stride=max_pool_scale)

    # Calc Vertical Horizontal gradient [B, H-1, W-1]
    gx = torch.zeros_like(logits)
    gy = torch.zeros_like(logits)
    gx[:, :, :-1] = logits.diff(dim=-1)
    gy[:, :-1, :] = logits.diff(dim=-2)

    # Store gradients of 8-connected neighbors (including the anchor pixel) into a tensor [B, 9, H-2, W-2]
    B, H, W = logits.size()
    gdMapx = torch.zeros((B, 9, H-2, W-2), device=logits.device, dtype=logits.dtype)
    gdMapy = torch.zeros((B, 9, H-2, W-2), device=logits.device, dtype=logits.dtype)

    for i in range(9):
        dx = i % 3
        dy = i // 3
        gdMapx[:, i, :, :] = gx[:, dy:dy+H-2, dx:dx+W-2]
        gdMapy[:, i, :, :] = gy[:, dy:dy+H-2, dx:dx+W-2]

    # Approximate L2 norm
    gdMap = gdMapx.pow(2) + gdMapy.pow(2)
    
    # Max gradient value [B, x, y]
    maxValue = gdMap.max(dim=1).values

    # Calc difference between maxValue & anchor pixel's gradient
    loss = torch.mean(torch.abs(gdMap[:,4,:,:] - maxValue))
    return loss

#####################################################
# Util functions ####################################
#####################################################

def softargmax(logits, dim=1):

    """
    Differentiable argmax
    https://gist.github.com/tejaskhot/fde6ca39209a2a6b6f1ebfad0d99f5ce
    """

    probs = torch.softmax(logits, dim=dim)  # logits [B,C,H,W], props [B,H,W]
    C = logits.size(dim)
    idx = torch.arange(C, device=logits.device, dtype=probs.dtype).view(1, C, 1, 1)
    return (probs * idx).sum(dim=dim) # [B,H,W]

@torch.no_grad()
def show_img(x):

    """
    Function for plotting
    Input: [H,W]
    """

    x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
    plt.imshow((x > 0).astype(np.float32), cmap="gray", vmin=0, vmax=1)
    plt.axis("off"); plt.show()