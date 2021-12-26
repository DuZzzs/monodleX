#!/usr/bin/env bash

CONFIG=$1


# python ../../tools/train_val.py --config ${CONFIG}
# ddp mode
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ../../tools/train_val.py \
--config test_kitti_train_dist.yaml --sync_bn
