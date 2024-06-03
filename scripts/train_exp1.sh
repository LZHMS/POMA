#!/bin/bash

DATASET=$1
TAG=$2

# Experiments: Training for POMA
# Configuration
# --- dataset: Dtd
# --- prompt blocks m: 1 | 2 | 4 | 6
# --- noise rate: 0 | 12.5% | 25% | 50%
# --- backbone: Text: ViT-B/32-PT, Visual: RN50-PT
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init${TAG} 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init${TAG} 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init${TAG} 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init${TAG} 8

CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init${TAG} 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init${TAG} 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init${TAG} 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init${TAG} 8

CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init${TAG} 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init${TAG} 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init${TAG} 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init${TAG} 8

CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init${TAG} 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init${TAG} 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init${TAG} 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init${TAG} 8

# Experiments: Result Analysis for POMA
# Configuration
# --- Experiments: Training for POMA
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init${TAG}
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init${TAG}
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init${TAG}
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init${TAG}