#!/usr/bin/env bash

CONFIG=$1

# train
#python ../../tools/train_val.py --config ${CONFIG}
#python ../../tools/train_val.py --config test_kitti.yaml
#python ../../tools/train_val.py --config test_kitti_res18.yaml
python ../../tools/train_val.py --config test_kitti_v2.yaml

# test
#python ../../tools/train_val.py --config test_kitti.yaml --e