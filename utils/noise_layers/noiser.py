import torch
import io
import numpy as np
import torch.nn as nn
from utils.noise_layers.identity import Identity
from utils.noise_layers.jpeg_compression import JpegCompression
from utils.noise_layers.noises import Rotation, CropandResize, GaussianBlur, GaussianNoise, ColorJitter, Sharpness, random_int
import kornia as K
import torchvision.transforms as T
from torchvision.io import ImageReadMode, read_image

class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list, posibilities: list , device):
        super(Noiser, self).__init__()
        self.noise_layers = [Identity()]
        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'Identity':
                    continue
                elif layer == 'Jpeg':
                    self.noise_layers.append(JpegCompression(device))
                elif layer == 'CropandResize':
                    self.noise_layers.append(CropandResize((256, 512), (256, 512)))
                elif layer == 'GaussianBlur':
                    self.noise_layers.append(GaussianBlur(10.0))
                elif layer == 'GaussianNoise':
                    self.noise_layers.append(GaussianNoise(0.2))
                elif layer == 'ColorJitter':
                    self.noise_layers.append(ColorJitter())
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().')
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)
        self.posibilities = posibilities

    def forward(self, encoded_and_cover, possibilites=None):
        # chose according to the probabilities
        random_noise_layer = np.random.choice(self.noise_layers, 1, p=self.posibilities if possibilites is None else possibilites)[0]
        return random_noise_layer(encoded_and_cover)

def distorsion_unit(encoded_image,type):
    """
    this modeule is used for robustness enhance fine-tuning,
    with lesser distorsions compared to the Noiser module used in pre-training
    """
    if type == 'color_jitter':
        distorted_image = K.augmentation.ColorJiggle(
                        brightness=(0.8,1.2),
                        contrast=(0.8,1.2),
                        saturation=(0.8,1.2),
                        hue=(-0.1,0.1),
                        p=1)(encoded_image)
    elif type == 'crop':
        crop_size_h = random_int(432, 512)
        crop_size_w = random_int(432, 512)
        distorted_image = T.RandomCrop(size=(crop_size_h, crop_size_w))(encoded_image)
        distorted_image = T.Resize(size=(512, 512),antialias=None)(distorted_image)
    elif type == 'blur':
        distorted_image = K.augmentation.RandomGaussianBlur((3, 5), (4.0, 4.0), p=1.)(encoded_image)
    elif type == 'noise':
        distorted_image = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(encoded_image)
        distorted_image = torch.clamp(distorted_image, 0, 1)
    else:
        raise ValueError(f'Wrong distorsion type.')

    return distorted_image
