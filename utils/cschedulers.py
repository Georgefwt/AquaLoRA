import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers import DDIMScheduler, DDPMScheduler, EulerDiscreteScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput, betas_for_alpha_bar
from diffusers.configuration_utils import register_to_config

class customDDPMScheduler(DDPMScheduler):

    def subtract_noise(
            self,
            noisy_samples: torch.FloatTensor,
            pred_noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=noisy_samples.device, dtype=noisy_samples.dtype)
        timesteps = timesteps.to(noisy_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        original_samples = (noisy_samples - sqrt_one_minus_alpha_prod * pred_noise) / sqrt_alpha_prod
        return original_samples

    def get_sqrt_alpha_prod_div_sqrt_one_minus_alpha_prod(
            self,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=timesteps.device)
        timesteps = timesteps

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        return sqrt_alpha_prod / sqrt_one_minus_alpha_prod

    def velocity_to_eplison(
        self,
        velocity_pred: torch.FloatTensor,
        noisy_model_input: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(device=timesteps.device)
        timesteps = timesteps

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        epsilon = sqrt_one_minus_alpha_prod[:,None,None,None] * noisy_model_input + sqrt_alpha_prod[:,None,None,None] * velocity_pred
        return epsilon
