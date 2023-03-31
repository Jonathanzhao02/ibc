#!/bin/bash

## Use "N" of the N-d environment as the arg

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/stack_trajectory/mlp_ebm_langevin_best.gin \
  --task=StackTrajectory-v0 \
  --tag=langevin_best \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='ibc/data/stack_trajectory/$stack_trajectory*.tfrecord'" \
