import os
import torch
import argparse
from utils_eval import simple_sample, simple_decode
import jsonlines
import glob

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

parser.add_argument("--msg_bits", type=int, default=48)
parser.add_argument("--msgdecoder", type=str, default=None)
parser.add_argument("--msg_gt", type=str, default=None)
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--tpr_threshold", type=float, default=1e-6)

args = parser.parse_args()

# read prompts frm prompts.txt
with open(args.prompt_path, 'r') as f:
    prompts = f.readlines()
    prompts = [prompt.strip() for prompt in prompts]
negative_prompt = ['out of frame'] * len(prompts)

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for seed in range(10):
    simple_sample(
        args.model,
        args.sampler,
        prompts,
        output_dir,
        lora=args.lora,
        lora_scale=args.lora_scale,
        negative_prompt=None,
        height=args.height,
        width=args.width,
        seed=[42+i+seed*100 for i in range(len(prompts))],
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        batch_size=1,
    )

if args.msgdecoder is not None:
    assert args.msg_gt is not None
    img_paths = sorted(glob.glob(f"{args.output_dir}/*.png"))
    simple_decode(
        args.msg_bits,
        args.msgdecoder,
        img_paths,
        msg_gt=args.msg_gt,
        resolution=args.resolution,
        tpr_threshold=args.tpr_threshold
    )

