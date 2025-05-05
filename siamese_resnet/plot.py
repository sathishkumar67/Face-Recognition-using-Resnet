from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Tuple
import torch

def show_triplet(triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], fig_size=(15, 5)) -> None:
    """Display a triplet of images
        torch.Tensor: A tensor representing an image in the format (C, H, W)
    args:
        triplet (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing three images (anchor, positive, negative)
    returns:
        None
    """
    # Get the anchor, positive, and negative images
    anchor, positive, negative = triplet['anchor'], triplet['positive'], triplet['negative']

    # Plot the anchor, positive, and negative images
    fig, ax = plt.subplots(1, 3, figsize=fig_size)
    ax[0].imshow(anchor.permute(1, 2, 0))
    ax[1].imshow(positive.permute(1, 2, 0))
    ax[2].imshow(negative.permute(1, 2, 0))
    ax[0].set_title('Anchor Image')
    ax[1].set_title('Positive Image')
    ax[2].set_title('Negative Image')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    plt.show()