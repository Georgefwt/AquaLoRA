import os
import torch
from dreamsim import dreamsim
from PIL import Image
import glob
import argparse
from tqdm import tqdm
import argparse
from utils_eval import simple_sample

def ds(model, preprocess, set1, set2):
    image_set1 = sorted(glob.glob(f'{set1}/*.png'))
    image_set2 = sorted(glob.glob(f'{set2}/*.png'))
    assert len(image_set1) == len(image_set2)
    distances = []
    for i in tqdm(range(len(image_set1))):
        img1 = preprocess(Image.open(image_set1[i])).to("cuda")
        img2 = preprocess(Image.open(image_set2[i])).to("cuda")
        distance = model(img1, img2) # The model takes an RGB image from [0, 1], size batch_sizex3x224x224
        distances.append(distance)
    print(torch.mean(torch.stack(distances)))
    return torch.mean(torch.stack(distances)).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--lora_scale", type=float, default=1.)
    parser.add_argument("--prompt_path", type=str, default="prompt.txt")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="output")

    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--sampler", type=str, default="dpms_m")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=7.5)

    args = parser.parse_args()

    with open(args.prompt_path, 'r') as f:
        prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]
    negative_prompt = ['out of frame'] * len(prompts)

    output_dir = args.output_dir
    os.makedirs(f"{output_dir}/w", exist_ok=True)
    os.makedirs(f"{output_dir}/nw", exist_ok=True)

    simple_sample(
        args.model,
        args.sampler,
        prompts,
        f"{output_dir}/nw",
        lora=None,
        lora_scale=args.lora_scale,
        negative_prompt=None,
        height=args.height,
        width=args.width,
        seed=[42+i for i in range(len(prompts))],
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        batch_size=1,
    )

    simple_sample(
        args.model,
        args.sampler,
        prompts,
        f"{output_dir}/w",
        lora=args.lora,
        lora_scale=args.lora_scale,
        negative_prompt=None,
        height=args.height,
        width=args.width,
        seed=[42+i for i in range(len(prompts))],
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        batch_size=1,
    )

    model, preprocess = dreamsim(pretrained=True)

    image_set1 = sorted(glob.glob(f'{output_dir}/w/*.png'))
    image_set2 = sorted(glob.glob(f'{output_dir}/nw/*.png'))
    assert len(image_set1) == len(image_set2)

    ds(model, preprocess, f'{output_dir}/w', f'{output_dir}/nw')

