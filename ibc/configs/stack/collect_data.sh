#!/bin/bash

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=2000 \
 --policy=oracle_stack \
 --task=Stack-v0 \
 --dataset_path=ibc/data/stack/stack.tfrecord \
 --replicas=2  \
 --use_image_obs=True
