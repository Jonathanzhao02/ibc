#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=40 \
 --policy=oracle_stack_lite_trajectory \
 --task=StackLiteTrajectory-v0 \
 --dataset_path=/data2/localuser/jjzhao/ibc/data/stack_lite_trajectory/stack_lite_trajectory.tfrecord \
 --replicas=1 \
 --use_image_obs=True
