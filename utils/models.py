import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn

from torch.utils.data import Dataset
import torch.nn.init as init
import torchvision.models.efficientnet as efficientnet
from torchvision.models.efficientnet import EfficientNet_B1_Weights
import lpips
import timm

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class Repeat(nn.Module):
    def __init__(self, *sizes):
        super(Repeat, self).__init__()
        self.sizes = sizes

    def forward(self, x):
        # We assume x has shape (N, C, H, W) and sizes is (H', W')
        return x.repeat(1, *self.sizes)

class SecretEncoder(nn.Module):
    def __init__(self, secret_len, base_res=32, resolution=64) -> None:
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.secret_len = secret_len
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_len, base_res*base_res),
            nn.SiLU(),
            View(-1, 1, base_res, base_res),
            Repeat(4, 1, 1),
            nn.Upsample(scale_factor=(2**(log_resolution-log_base), 2**(log_resolution-log_base))),  # chx16x16 -> chx256x256
            zero_module(conv_nd(2, 4, 4, 3, padding=1))
        )  # secret len -> ch x res x res
    
    def copy_encoder_weight(self, ae_model):
        # misses, ignores = self.load_state_dict(ae_state_dict, strict=False)
        return None

    def encode(self, x):
        x = self.secret_scaler(x)
        return x
    
    def forward(self, x, c):
        # x: [B, C, H, W], c: [B, secret_len]
        c = self.encode(c)
        c = F.interpolate(
            c, size=(x.shape[2], x.shape[3]), mode='bilinear'
        )
        x = x + c
        return x, c


class SecretDecoder(nn.Module):
    def __init__(self, output_size=64):
        super(SecretDecoder, self).__init__()
        self.output_size=output_size
        self.model = efficientnet.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, output_size*2, bias=True)

    def forward(self, x):
        x = F.interpolate(
            x, size=(512, 512), mode='bilinear'
        )
        decoded = self.model(x).view(-1, self.output_size, 2)
        return decoded

class MapperNet(nn.Module):
    def __init__(self, input_size=16, output_size=64, std=1.):
        super(MapperNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bit_embeddings = nn.Embedding(
            input_size, output_size
        )
        init.orthogonal_(self.bit_embeddings.weight)
        self.bit_embeddings.weight.data = self.bit_embeddings.weight.data / self.bit_embeddings.weight.data.std(dim=1, keepdim=True)
        self.bit_embeddings.weight.data = self.bit_embeddings.weight.data * std

    def forward(self, x):
        pos_idx = torch.arange(self.input_size).long().to(x.device)
        encoded = self.bit_embeddings(pos_idx) # [48,224]
        encoded = encoded * x[:, :, None] # [4,48,224]
        encoded = encoded.sum(dim=1) / torch.sqrt(torch.tensor(self.input_size).float()) + 1.
        return encoded
