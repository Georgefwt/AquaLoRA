# conversion script for convert diffusers lora to A1111 webui

import re

def diffuers2webui(args):
    newDict = dict()
    for idx, key in enumerate(checkpoint):
        print(idx, key)
        newKey = re.sub('\.processor\.', '_', key)
        newKey = re.sub('mid_block\.', 'mid_block_', newKey)
        newKey = re.sub('_lora.up.', '.lora_up.', newKey)
        newKey = re.sub('.lora.up.', '.lora_up.', newKey)
        newKey = re.sub('_lora.down.', '.lora_down.', newKey)
        newKey = re.sub('.lora.down.', '.lora_down.', newKey)
        newKey = re.sub('\.(\d+)\.', '_\\1_', newKey)
        newKey = re.sub('_lora_up.', '.lora_up.', newKey)
        newKey = re.sub('_lora_down.', '.lora_down.', newKey)
        newKey = re.sub('to_out', 'to_out_0', newKey)
        newKey = re.sub('unet.', 'lora_unet_', newKey)
        newKey = re.sub('_ff.net_', '_ff_net_', newKey)

        newDict[newKey] = checkpoint[key]

    return newDict


if __name__ == "__main__":
    import torch
    import safetensors
    from safetensors.torch import save_file
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--src_lora', type=str, default=None)
    argparser.add_argument('--tgt_lora', type=str, default=None)
    args = argparser.parse_args()

    checkpoint = safetensors.torch.load_file(args.src_lora,device='cpu')

    webuilora = diffuers2webui(checkpoint)
    print("Saving " + args.tgt_lora)
    save_file(webuilora, args.tgt_lora)

