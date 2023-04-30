#!/bin/bash

## Use "N" of the N-d environment as the arg

python3 ibc/ibc/train_eval.py -- \
  --alsologtostderr \
  --gin_file=ibc/ibc/configs/stack_lite_trajectory/mlp_ebm_langevin_best.gin \
  --task=StackLiteTrajectory-v0 \
  --tag=langevin_best \
  --add_time=True \
  --gin_bindings="train_eval.dataset_path='/data2/localuser/jjzhao/ibc/data/stack_lite_trajectory_staticpos/$stack_lite_trajectory_staticpos*.tfrecord'" \
  --gin_bindings="StackLiteTrajectoryEnv.random_place=False" \
