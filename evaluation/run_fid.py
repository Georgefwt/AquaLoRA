# Modified from https://github.com/YuxinWenRick/tree-ring-watermark/blob/main/run_tree_ring_watermark_fid.py

import torch
import argparse
import copy
from tqdm import tqdm
import json

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from pytorch_fid.fid_score import *


def main(args):
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        safety_checker=None
        )
    pipe = pipe.to(device)

    if args.lora is not None:
        pipe.load_lora_weights(args.lora, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(lora_scale=1.)

    with open(args.prompt_file) as f:
        dataset = json.load(f)
        image_files = dataset['images']
        dataset = dataset['annotations']
        prompt_key = 'caption'

    w_dir = f'fid_outputs/coco/{args.run_name}/w_gen'
    os.makedirs(w_dir, exist_ok=True)

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]
        
        ### generation
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            generator=torch.Generator(device=device).manual_seed(seed),
            )
        orig_image_w = outputs_w.images[0]

        image_file_name = image_files[i]['file_name']
        orig_image_w.save(f'{w_dir}/{image_file_name}')

    ### calculate fid
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    # fid for w
    fid_value_w = calculate_fid_given_paths([args.gt_folder, w_dir],
                                          50,
                                          device,
                                          2048,
                                          num_workers)

    print(f'fid_w: {fid_value_w}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=5000, type=int)
    parser.add_argument('--lora', type=str, default=None)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--run_no_w', action='store_true')
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--prompt_file', default='coco/meta_data.json')
    parser.add_argument('--gt_folder', default='coco/ground_truth')

    args = parser.parse_args()

    main(args)
