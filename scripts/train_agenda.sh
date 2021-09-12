#!/bin/bash

python -u finetune.py \
    --data_dir dataset/agenda_baseline \
    --model_name_or_path facebook/bart-base  \
    --output_dir outputs/agenda_out \
    --max_source_length 100 \
    --max_target_length 200 \
    --val_max_target_length 200 \
    --num_train_epochs 25  \
    --learning_rate 3e-5 \
    --fp16 \
    --do_train \
    --do_eval \
    --eval_beams 3 \
    --per_device_train_batch_size 35 \
    --per_device_eval_batch_size 40 \
    --predict_with_generate \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --evaluate_during_training