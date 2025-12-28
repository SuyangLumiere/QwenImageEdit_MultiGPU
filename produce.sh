#! /bin/bash
MODEL_PATH="/rico_model/qwen_image_edit"
IMG="output/"
CTRL="input/"
CACHE="cache/"

RESOLUTION=$((512*512))

python scripts/producer.py \
    --pretrained_model "$MODEL_PATH" \
    --img_dir "$IMG" \
    --control_dir "$CTRL" \
    --target_area $RESOLUTION \
    --output_dir "$CACHE" \
    --prompt_with_image \
    "$@"