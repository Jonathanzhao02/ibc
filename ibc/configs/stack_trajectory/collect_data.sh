#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --policy=oracle_stack_trajectory \
 --task=StackTrajectory-v0 \
 --dataset_path=ibc/data/stack_trajectory/stack_trajectory.tfrecord \
 --replicas=1  \
 --use_image_obs=True
