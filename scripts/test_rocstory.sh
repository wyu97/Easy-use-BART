#!/bin/bash

MODEL_PATH=MODEL_FROM_BEST_DEV_EPOCH

python -u ../finetune.py \
    --data_dir ../dataset/rocstory_baseline \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ../outputs/rocstory_out \
    --max_source_length 40 \
    --max_target_length 150 \
    --val_max_target_length 150 \
    --fp16 \
    --do_predict \
    --per_device_eval_batch_size 60 \
    --eval_beams 3 \
    --predict_with_generate \
    --overwrite_output_dir \
    # --do_sample \
    # --top_k 0 \
    # --top_p 1.0 \