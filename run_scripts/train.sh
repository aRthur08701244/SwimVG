#!/bin/bash
dataset_name="refcoco+" 
config_name="swimvg_dinov2.yaml"


np=1
omp=16

CUDA_VISIBLE_DEVICES=7 \
MASTER_PORT=29560 \
OMP_NUM_THREADS=$omp \

torchrun --nproc_per_node=$np --master_port=$MASTER_PORT ./train.py \
--config config/$dataset_name/$config_name \
--aug_crop --aug_scale --aug_translate \
