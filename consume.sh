#! /bin/bash
MODEL_PATH="/rico_model/qwen_image_edit"
TXT="cache/text_embs/"
IMG="cache/img_embs/"
CTRL="img_embs_control/"
OUTPUT="result/"

LORA_RANK=64
LR=3e-4

EPOCH=1000
WARM_STEP=300
MAX_STEP=5000
CKP=500

python scripts/pp_consumer.py \
    --output_dir "$OUTPUT" \
    --logging_dir "./logger" \
    --pretrained_model "$MODEL_PATH" \
    --rank $LORA_RANK \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --epochs $EPOCH \
    --max_train_steps $MAX_STEP \
    --lr_warmup_steps $WARM_STEP \
    --lr_scheduler constant_with_warmup \
    --learning_rate $LR \
    --num_workers 4 \
    --max_grad_norm 1.0 \
    --checkpointing_steps $CKP \
    --txt_cache_dir "$TXT" \
    --img_cache_dir "$IMG" \
    --control_img_cache_dir "$CTRL" \
    "$@"