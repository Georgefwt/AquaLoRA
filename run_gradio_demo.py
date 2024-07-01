import sys
sys.path.append("./scripts")
import gradio as gr
import numpy as np
from scripts.create_wm_lora import create_watermark_lora
from scripts.webui_lora_to_diffusers import webui2diffusers
from scripts.merge_lora import merge
from evaluation.utils_eval import simple_sample, simple_decode

def process(src_model, aqualora_model, secret, prompt, n_prompt, ddim_steps, cfg, seed):
    _, lora_state_dict = create_watermark_lora(aqualora_model, scale=1.03, msg_bits=48, hidinfo=secret, save=False)

    images = simple_sample(src_model,"ddim",[prompt],
                output_dir=None,
                lora=lora_state_dict,
                negative_prompt=[n_prompt],
                num_inference_steps=int(ddim_steps),
                guidance_scale=cfg,
                seed=seed,
                save=False,
                )

    bitacc, TPR, results = simple_decode(48,
                f"{aqualora_model}/msgdecoder.pt",
                images,
                msg_gt=secret
                )

    return images, results[0]


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## AquaLoRA Demo")
    with gr.Row():
        with gr.Column():
            src_model = gr.Textbox(label="clean model distination")
            aqualora_model = gr.Textbox(label="AquaLoRA model distination", value="AquaLoRA-Models/ppft_trained")
            secret = gr.Textbox(label="watermark secret", value='110110010011010110001111010101010101011000100110')
            prompt = gr.Textbox(label="Prompt", value='a parade with cars and people waving')
            prompt_examples = gr.Examples(
                examples=[
                    "hyper-realistic photo, bird in the wild",
                    "A combat ready female cyborg, neon, downtown tokyo background"
                ],
                inputs=prompt)
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                cfg = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=15.0, value=7.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=999999999, step=1, value=625078911)
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=True, elem_id="gallery", columns=[2], rows=[1], object_fit="contain", height="auto")
            decoded_secret = gr.Textbox(label="Decoded Secret", placeholder="Decoded secret will appear here")
    ips = [src_model, aqualora_model, secret, prompt, n_prompt, ddim_steps, cfg, seed]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery, decoded_secret])


block.launch()
