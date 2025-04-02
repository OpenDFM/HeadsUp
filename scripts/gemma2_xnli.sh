#!/bin/bash

LM_MODEL_PATH="~/PretrainedModels/gemma-2-2b-it"
TRAIN_DATA_PATH="../dataset/XNLI-15way/xnli.15way.orig.tsv"
DEV_DATA_PATH="../dataset/XNLI-15way/xnli.15way.orig.tsv"
NUM_GPUS=1

source activate modularity
cd ../

WANDB_MODE=offline python train_mask.py \
    --lm_model_path $LM_MODEL_PATH \
    --train_data_path $TRAIN_DATA_PATH \
    --dev_data_path $DEV_DATA_PATH \
    --model_type gemma2 \
    --seed 42 \
    --dataset_use_cache false \
    --output_dir ./output/gemma2/xnli \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --remove_unused_columns false \
    --save_strategy steps \
    --save_steps 31 \
    --save_total_limit 200 \
    --save_only_model \
    --evaluation_strategy epoch \
    --logging_dir ./logs \
    --logging_steps 100 \
    --learning_rate 1e-2 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-4}' \
    --tf32 true \
    --ddp_find_unused_parameters false \
    --weight_decay 0 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 1 \
    --use_gumbel \
    --gumbel_hard \
    --tau_temp_begin 4 \
    --tau_decay_steps 0.4 \
    --tau_temp_end 0.05 \
    --norm_lambda 0 \
    --norm_power 1 \
    --report_to none \