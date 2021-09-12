#!/bin/bash

python -u finetune.py \
    --data_dir dataset/rocstory_baseline \
    --model_name_or_path facebook/bart-base  \
    --output_dir outputs/rocstory_out \
    --max_source_length 40 \
    --max_target_length 150 \
    --val_max_target_length 150 \
    --num_train_epochs 25  \
    --learning_rate 3e-5 \
    --fp16 \
    --do_train \
    --do_eval \
    --eval_beams 3 \
    --per_device_train_batch_size 40 \
    --per_device_eval_batch_size 50 \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --evaluate_during_training