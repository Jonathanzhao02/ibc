#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --policy=oracle_stack_trajectory \
 --task=StackTrajectory-v0 \
 --output_path="/data2/localuser/jjzhao/ibc/misc" \
 --viz_img=True \
 --use_image_obs=True
