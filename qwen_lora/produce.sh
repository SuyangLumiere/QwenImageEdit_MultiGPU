#! /bin/bash
python producer.py \
    --pretrained_model "qwen_image_edit" \
    --img_dir "output" \
    --control_dir "input" \
    --target_area 512*512 \
    --output_dir "cache" \
    --prompt_with_image