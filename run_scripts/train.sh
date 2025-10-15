#!/bin/bash
dataset_name="refcoco+" 
config_name="swimvg_dinov2.yaml"

gpu="0" #"7"
np=$(echo $gpu | tr -cd ',' | wc -c)
np=$((np + 1))
omp=8

MASTER_PORT=29560

OMP_NUM_THREADS=$omp \
CUDA_VISIBLE_DEVICES=$gpu \
torchrun --nproc_per_node=$np --master_port=$MASTER_PORT ./train.py \
--config config/$dataset_name/$config_name \
--aug_crop --aug_scale --aug_translate \
