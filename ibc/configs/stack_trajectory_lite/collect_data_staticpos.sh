#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=5 \
 --policy=oracle_stack_lite_trajectory \
 --task=StackLiteTrajectory-v0 \
 --dataset_path=ibc/data/stack_trajectory_lite_staticpos/stack_trajectory_lite_staticpos.tfrecord \
 --replicas=1 \
 --gin_bindings="StackLiteTrajectoryEnv.random_place=False" \
 --use_image_obs=True
