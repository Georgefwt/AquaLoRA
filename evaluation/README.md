## Evaluation of AquaLoRA

Before starting the test, you need to convert the trained `mapper.pt`, `msgdecoder.pt`, and `pytorch_lora_weights.safetensors` to obtain a regular LoRA. At this stage, you can choose the secret message you want. You can use the following command to perform the conversion:

```bash
cd scripts
python create_wm_lora.py --train_folder <your train folder> --hidinfo "1010..." (when hidinfo is none, script will chose a random bit string)
```

The organization format of the train folder is as follows:

```
<your train folder>
├── mapper.pt
├── msgdecoder.pt
└── pytorch_lora_weights.safetensors
```

After the command is executed, a LoRA will be generated in the train folder:

```
<your train folder>
├── 110110011110000111011100001011100100011001110001 <secret message>
│   └── pytorch_lora_weights.safetensors
├── mapper.pt
├── msgdecoder.pt
└── pytorch_lora_weights.safetensors
```

### Test True Positive Rate and Bit Accuracy

For testing AquaLoRA with no distortion, you can run,

```bash
python run_eval_base.py --lora <your train folder>/<secret message> --msgdecoder <your train folder>/msgdecoder.pt --msg_gt <secret message>
```

To test the performance of AquaLoRA under noise perturbation, you can run,

```bash
python run_eval_distortion.py --lora <your train folder>/<secret message> --msgdecoder <your train folder>/msgdecoder.pt --msg_gt <secret message>
```

### Calculate DreamSim Score

To calculate the [DreamSim](https://github.com/ssundaram21/dreamsim) Score, you can run,

```bash
python run_dreamsim.py --lora <your train folder>/<secret message> --output_dir output
```

### Calculate FID

When calculating FID, we referred to [Tree-Ring Watermark](https://github.com/YuxinWenRick/tree-ring-watermark), using the COCO2017 validation set, which contains a total of 5000 images. You can find the corresponding information such as prompts in 'fid_outputs/coco/meta_data.json'.

Then, to calculate FID, you can run,

```bash
python run_fid.py --lora <your train folder>/<secret message> --prompt_file fid_outputs/coco/meta_data.json --gt_folder fid_outputs/coco/ground_truth
```
