# conversion script for convert A1111 webui lora to diffusers

import re

def webui2diffusers(checkpoint):
    newDict = {}
    index = 0
    for idx, key in enumerate(checkpoint):
        omitted_keys = ['_te_text_', '_ff_net_', 'alpha', '_proj_']
        if any([x in key for x in omitted_keys]):
            continue
        newKey = re.sub('^lora_unet_down_blocks_', 'down_blocks.', key)
        newKey = re.sub('^lora_unet_up_blocks_', 'up_blocks.', newKey)
        newKey = re.sub('^lora_unet_mid_block_', 'mid_block.', newKey)
        newKey = re.sub('_attentions_', '.attentions.', newKey)
        newKey = re.sub('_transformer_blocks_', '.transformer_blocks.', newKey)
        newKey = re.sub('_attn(\d+)_', '.attn\\1.processor.', newKey)
        newKey = re.sub('_to_', '.to_', newKey)
        newKey = re.sub('\.lora_up\.', '_lora.up.', newKey)
        newKey = re.sub('\.lora_down\.', '_lora.down.', newKey)
        newKey = re.sub('_0', '', newKey)
        newKey = re.sub('_alpha', '.alpha', newKey)
        newKey = re.sub('mid_block.attentions.', 'mid_block.attentions.0.', newKey)
        newDict[newKey] = checkpoint[key]
        index += 1
        print(index, newKey)
    return newDict

if __name__ == "__main__":
    import torch
    import safetensors
    from safetensors.torch import save_file
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lora", type=str, required=True)
    parser.add_argument("--tgt_lora", type=str, required=True)
    args = parser.parse_args()

    checkpoint = safetensors.torch.load_file(args.src_lora,device='cpu')

    diffuserlora = webui2diffusers(checkpoint)
    print("Saving " + args.tgt_lora)
    save_file(diffuserlora, args.tgt_lora)
