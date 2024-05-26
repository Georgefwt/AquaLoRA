## Training of AquaLoRA

Here is a simple discription to the training process of AquaLoRA. In short, the training of AquaLoRA is divided into latent watermark pre-training, prior preserving fine-tuning, and robustness enhancement (optional). Our training setup is based on the A6000 40GB GPU, so you may need to adjust the settings according to your hardware environment.

### Latent Watermark Pre-training

To train the watermark on the latent space, you need to download the [COCO2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) dataset. Then, copy the `train/dataset_assets/metadata.jsonl` file we provided into the target dataset directory. Ensure the directory structure is as follows:

```
COCO2017test
└── train
    ├── 000000000071.jpg
    ├── other jpgs...
    └── metadata.jsonl
```

Then run the following command to start the training:

```bash
python latent_wm_pretrain.py --output_dir output1 --epochs 40 --dataset COCO2017test
```

After completing the training, you can choose an appropriate checkpoint and rename it to `pretrained_latentwm.pth` for use in the next training stage.

### Prior Preserving Fine-tuning

To perform Prior Preserving Fine-Tuning (PPFT), you first need to download the dataset from [here](https://huggingface.co/datasets/georgefen/Gustavosta-sample) and extract data from compressed file. Then, execute the following command to start the training:


```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export TRAIN_DIR="Gustavosta-sample"

accelerate launch --mixed_precision="fp16" ppft_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR --caption_column="text" \
  --resolution=512 \
  --dataloader_num_workers=12 \
  --train_batch_size=12 \
  --num_train_epochs=30 --checkpointing_steps=1000 \
  --learning_rate=1e-04 --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=0 --lr_end=0.01 \
  --seed=2048 \
  --output_dir="output2" \
  --start_from_pretrain="pretained_latentwm.pth" \
  --validation_prompt="A portrait of a young white woman, masterpiece" \
  --validation_epochs=1 \
  --rank=320 \
  --msg_bits=48 \
```

After the training is completed, you will obtain three model files: `mapper.pt`, `msgdecoder.pt`, and `pytorch_lora_weights.safetensors`.

### Robustness Enhancement (Optional)

As mentioned in our paper, we further freeze all weights except the decoder and fine-tune the decoder to enhance the watermark extraction capability for larger sampling sizes. We continue to use the previously used COCO2017 dataset. Execute the following command to start the training:

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export TRAIN_DIR="COCO2017test"

accelerate launch --mixed_precision="fp16" rob_enhance_finetune.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR --caption_column="text" \
  --resolution=512 \
  --dataloader_num_workers=16 \
  --train_batch_size=16 \
  --num_train_epochs=10 --checkpointing_steps=200 \
  --learning_rate=5e-06 --lr_scheduler="cosine_with_restarts" --lr_warmup_steps=0 --lr_end=0.1 \
  --seed=65535 \
  --output_dir="output3" \
  --start_from_pretrain="pretained_latentwm.pth" \
  --resume_from_lora="output2" \
  --validation_prompt="A portrait of a young white woman, masterpiece" \
  --validation_epochs=1 \
  --rank=320 \
  --msg_bits=48
```

