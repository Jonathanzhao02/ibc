#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --policy=oracle_stack_lite_trajectory \
 --task=StackLiteTrajectory-v0 \
 --output_path="/data2/localuser/jjzhao/ibc/imgs" \
 --viz_img=True \
 --use_image_obs=True
