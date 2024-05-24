from PIL import Image
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math

def torch_to_pil(images):
    images = images.detach().cpu().float()
    if images.ndim == 3:
        images = images[None, ...]
    images = images.permute(0, 2, 3, 1)
    images = (images + 1) * 0.5
    images = (images * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def get_cosine_schedule_with_warmup_lr_end(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, lr_end=0.0
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(lr_end, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

