import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import kornia as K
import numpy as np

def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min

def random_int(min, max):
    return np.random.randint(min, max)

class Rotation(nn.Module):
    """
    Rotates the image by a random angle between 0 and 180 degrees
    """
    def __init__(self, degrees=180):
        super(Rotation, self).__init__()
        self.degrees = degrees

    def forward(self, noised_and_cover):
        distorted_image = K.augmentation.RandomRotation(degrees=self.degrees, p=1)(noised_and_cover[0])
        noised_and_cover[0] = distorted_image
        return noised_and_cover


class CropandResize(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, crop_size_range, resize_size_range):
        super(CropandResize, self).__init__()
        self.crop_size_min = crop_size_range[0]
        self.crop_size_max = crop_size_range[1]
        self.resize_size_min = resize_size_range[0]
        self.resize_size_max = resize_size_range[1]

    def forward(self, noised_and_cover):
        crop_size_h = random_int(self.crop_size_min, self.crop_size_max)
        crop_size_w = random_int(self.crop_size_min, self.crop_size_max)
        resize_size_h = random_int(self.resize_size_min, self.resize_size_max)
        resize_size_w = random_int(self.resize_size_min, self.resize_size_max)

        distorted_image = T.RandomCrop(size=(crop_size_h, crop_size_w))(noised_and_cover[0])
        distorted_image = T.Resize(size=(resize_size_h, resize_size_w),antialias=None)(distorted_image)
        distorted_image = T.Resize(size=(512, 512),antialias=None)(distorted_image)
        noised_and_cover[0] = distorted_image

        return noised_and_cover

class GaussianBlur(nn.Module):
    """
    Blurs the image with a gaussian noise
    """
    def __init__(self, blur=2.0):
        super(GaussianBlur, self).__init__()
        self.gaussian_blur_max = blur

    def forward(self, noised_and_cover):
        distorted_image = K.augmentation.RandomGaussianBlur((3, 9), (0, self.gaussian_blur_max), p=1.)(noised_and_cover[0])
        noised_and_cover[0] = distorted_image
        return noised_and_cover

class GaussianNoise(nn.Module):
    """
    Adds gaussian noise to the image
    """
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.gaussian_std_max = std

    def forward(self, noised_and_cover):
        gaussian_std = random_float(0, self.gaussian_std_max)
        # add noise
        distorted_image = K.augmentation.RandomGaussianNoise(mean=0.0, std=gaussian_std, p=1)(noised_and_cover[0])
        noised_and_cover[0] = distorted_image
        return noised_and_cover


class ColorJitter(nn.Module):
    """
    Jitters the color of the image
    """
    def __init__(self):
        super(ColorJitter, self).__init__()

    def forward(self, noised_and_cover):
        norm_img = noised_and_cover[0] / 2 + 0.5
        distorted_image = K.augmentation.ColorJiggle(
            brightness=(0.7,1.3),
            contrast=(0.8,1.25),
            saturation=(0.8,1.25),
            hue=(-0.2,0.2),
            p=1)(norm_img)
        noised_and_cover[0] = distorted_image * 2 - 1
        return noised_and_cover

class Sharpness(nn.Module):
    """
    Sharpens the image
    """
    def __init__(self, strength=1.):
        super(Sharpness, self).__init__()
        self.strength_max = strength

    def forward(self, noised_and_cover):
        strength = random_float(0, self.strength_max)
        norm_img = noised_and_cover[0] / 2 + 0.5
        distorted_image = K.augmentation.RandomSharpness(sharpness=strength, p=1)(norm_img)
        noised_and_cover[0] = distorted_image * 2 - 1
        return noised_and_cover