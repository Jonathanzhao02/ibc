#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=2 \
 --policy=oracle_stack_trajectory \
 --task=StackTrajectory-v0 \
 --dataset_path=/data2/localuser/jjzhao/ibc/data/stack_trajectory/stack_trajectory.tfrecord \
 --replicas=1 \
 --use_image_obs=True
