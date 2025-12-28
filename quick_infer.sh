#! /bin/bash

MODEL_PATH="/rico_model/qwen_image_edit"
LORA_PATH="./checkpoint-500"
INPUT_IMG="./test_data/input_001.png"
OUTPUT_DIR="./outputs"

RESOLUTION=$((512*512))

python scripts/quick_infer.py \
    --pretrained_model "$MODEL_PATH" \
    --lora_weight "$LORA_PATH" \
    --ctrl_img "$INPUT_IMG" \
    --output_img "$OUTPUT_DIR/result_$(date +%Y%m%d_%H%M%S).png" \
    --prompt "Place a detailed action figure standing upright on the snowy slope" \
    --cfg_scale 6.0 \
    --infer_steps 50 \
    --target_area $RESOLUTION \
    "$@"