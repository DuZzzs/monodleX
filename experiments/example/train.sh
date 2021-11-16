#!/usr/bin/env bash

CONFIG=$1

# train
#python ../../tools/train_val.py --config ${CONFIG}

# test
python ../../tools/train_val.py --config test_kitti.yaml --e