import os
import sys
sys.path.append("../")
import torch
import io
from math import comb
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.efficientnet as efficientnet
from torchvision.models.efficientnet import EfficientNet_B1_Weights
from torchvision.io import ImageReadMode, read_image
from diffusers import (
    DDPMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DPMSolverSDEScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    AutoencoderKL,
    # ConsistencyDecoderVAE
    StableDiffusionImg2ImgPipeline,
)
from typing import List, Optional, Tuple, Union
import kornia as K
import PIL.Image as Image
import numpy as np
from tqdm import tqdm

def simple_sample(
        model,
        sampler,
        prompt: List[str],
        output_dir: Union[str, List[str]],
        lora=None,
        lora_scale=1.,
        negative_prompt: Optional[List[str]] = None,
        height: Union[int, List[int]] = 512,
        width: Union[int, List[int]] = 512,
        seed: Union[int, List[int]] = 42,
        num_inference_steps: Union[int, List[int]] = 50,
        guidance_scale: Union[float, List[float]] = 7.5,
        batch_size: int = 1,
):
    if negative_prompt is not None:
        assert len(negative_prompt) == len(prompt)
        if not isinstance(seed, list):
            seed = [seed] * len(prompt)
        else:
            assert len(prompt) == len(seed)

    def transform_to_list(var, batch_size, prompt):
        if not isinstance(var, list):
            return [var] * (len(prompt) // batch_size)
        else:
            assert len(prompt) == len(var) * batch_size
            return var

    height = transform_to_list(height, batch_size, prompt)
    width = transform_to_list(width, batch_size, prompt)
    num_inference_steps = transform_to_list(num_inference_steps, batch_size, prompt)
    guidance_scale = transform_to_list(guidance_scale, batch_size, prompt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model.endswith(".ckpt") or model.endswith(".safetensors"):
        pipe = StableDiffusionPipeline.from_single_file(model, load_safety_checker=False)
    else:
        # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        # vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder")
        pipe = StableDiffusionPipeline.from_pretrained(model, safety_checker=None)

    if lora is not None:
        pipe.load_lora_weights(lora, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(lora_scale=lora_scale)
    if sampler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "heun":
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "lms":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "pndm":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "dpms_s":
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    elif sampler == "dpms_sde":
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config)
    elif sampler == "dpms_m":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif sampler == "kdpm2":
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "kdpm2a":
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Sampler {sampler} not found.")
    pipe.to(device)

    generator = [torch.Generator(device=device).manual_seed(seed[i]) for i in range(len(seed))]

    for i in range(len(prompt)//batch_size):
        images = pipe(
                prompt[i*batch_size:(i+1)*batch_size],
                negative_prompt=negative_prompt[i*batch_size:(i+1)*batch_size] if negative_prompt is not None else None,
                height=height[i],
                width=width[i],
                num_inference_steps=num_inference_steps[i],
                guidance_scale=guidance_scale[i],
                generator = generator[i*batch_size:(i+1)*batch_size]
                ).images
        for j,img in enumerate(images):
            if isinstance(output_dir, list):
                img.save(f"{output_dir[i]}")
            else:
                img.save(f"{output_dir}/{seed[i]}_{j}.png")

# --------------------------------------------------------------------------

def calculate_fpr(tau, k):
    sum_combinations = sum(comb(k, i) for i in range(tau + 1, k + 1))
    fpr = 1 / (2**k) * sum_combinations
    return fpr

def get_threshold(k, fpr):
    tau = 0
    while calculate_fpr(tau, k) > fpr:
        tau += 1
    return tau

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

def simple_decode(
        bitnum,
        msgdecoder_path,
        img_paths: List[str],
        msg_gt = None,
        resolution: int = 512,
        tpr_threshold: float = 1e-3
):
    msgdecoder = SecretDecoder(output_size=bitnum)
    msgdecoder.load_state_dict(torch.load(msgdecoder_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    msgdecoder = msgdecoder.to(device)
    msgdecoder.eval()

    tau = get_threshold(bitnum, tpr_threshold) / bitnum

    def process(image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

    results = []
    accuracy = []
    TP = 0
    FN = 0
    for img_path in tqdm(img_paths):
        image = Image.open(img_path)
        image = process(image)
        image = image.cuda()
        image = image.unsqueeze(0)

        # decode the message
        msg = msgdecoder(image)
        msg = torch.argmax(msg, dim=-1)
        msg = ''.join(map(str, msg.tolist()[0]))
        results.append(msg)
        # calculate bit accuracy between msg_gt and msg
        if msg_gt is not None:
            assert len(msg_gt) == len(msg)
            acc = sum([1 for i in range(len(msg)) if msg[i] == msg_gt[i]]) / len(msg)
            accuracy.append(acc)
            if acc >= tau:
                TP += 1
            else:
                FN += 1
    if msg_gt is not None:
        bitacc = np.mean(accuracy)
        TPR = TP / (TP + FN)
        print(f"bit accuracy: {bitacc}")
        print(f"TPR: {TPR}")

    return bitacc, TPR, results


# --------------------------------------------------------------------------

vae = None
pipe = None
pipe2 = None

def resize_decorator(distortion_func):
    def wrapper(encoded_image, *args, **kwargs):
        original_size = encoded_image.shape[2:]  # Save original size
        if original_size != (512, 512):
            encoded_image = T.Resize(size=(512, 512))(encoded_image)
        # Apply the distortion
        distorted_image = distortion_func(encoded_image, *args, **kwargs)
        return distorted_image
    return wrapper

def torch_to_pil(images):
    images = images.detach().cpu().float()
    if images.ndim == 3:
        images = images[None, ...]
    images = images.permute(0, 2, 3, 1)
    images = (images * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def SDEdit(image, version=1):
    global pipe
    global pipe2
    if version == 1:
        if pipe is None:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', safety_checker=None).to('cuda')
        cur_pipe = pipe
    elif version == 2:
        if pipe2 is None:
            pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', safety_checker=None).to('cuda')
        cur_pipe = pipe2
    oimages = torch_to_pil(image)
    images = cur_pipe(prompt="masterpiece", 
                  image=oimages,
                  strength=0.1 if version == 1 else 0.2,
                  num_inference_steps=10,
                  guidance_scale=7.5 ,
                  output_type='pt').images
    return images.squeeze(0)

@resize_decorator
def crop(encoded_image,size=(460, 460)):
    distorted_image = T.RandomCrop(size=size)(encoded_image)
    return distorted_image

def distorsion_unit(encoded_image,type):
    if type == 'color_jitter':
        distorted_image = K.augmentation.ColorJiggle(
                        brightness=(0.9,1.1),
                        contrast=(0.9,1.1),
                        saturation=(0.9,1.1),
                        hue=(-0.1,0.1),
                        p=1)(encoded_image)
    elif type == 'crop':
        distorted_image = crop(encoded_image)
    elif type == 'blur':
        distorted_image = K.augmentation.RandomGaussianBlur((3, 3), (4.0, 4.0), p=1.)(encoded_image)
    elif type == 'noise':
        distorted_image = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(encoded_image)
        distorted_image = torch.clamp(distorted_image, 0, 1)
    elif type == 'jpeg_compress':
        buffer = io.BytesIO()
        out = T.ToPILImage()(encoded_image.squeeze(0))
        out.save(buffer, format='JPEG', quality=50)
        buffer.seek(0)
        pil_image = Image.open(buffer)
        distorted_image = T.ToTensor()(pil_image)
    elif type == 'rotation':
        distorted_image = K.augmentation.RandomRotation(degrees=(15,15), p=1)(encoded_image)
    elif type == 'sharpness':
        distorted_image = K.augmentation.RandomSharpness(sharpness=10., p=1)(encoded_image)
    elif type == 'SDEdit':
        distorted_image = SDEdit(encoded_image)
    elif type == 'SDEdit2':
        distorted_image = SDEdit(encoded_image,version=2)
    else:
        raise ValueError(f'Wrong distorsion type in add_distorsion().')
    return distorted_image

def apply_distorsion(input_image_path, output_image_path, type):
    for path in tqdm(input_image_path):
        encoded_image = read_image(path, ImageReadMode.RGB).to(torch.float32) / 255
        encoded_image = encoded_image.cuda()
        distorted_image = distorsion_unit(encoded_image, type)

        distorted_image_path = os.path.join(output_image_path, type, os.path.basename(path))
        out = T.ToPILImage()(distorted_image.squeeze(0))
        out.save(distorted_image_path)


