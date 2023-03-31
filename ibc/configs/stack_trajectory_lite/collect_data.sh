#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=500 \
 --policy=oracle_stack_lite_trajectory \
 --task=StackLiteTrajectory-v0 \
 --dataset_path=ibc/data/stack_trajectory_lite/stack_trajectory_lite.tfrecord \
 --use_image_obs=True
