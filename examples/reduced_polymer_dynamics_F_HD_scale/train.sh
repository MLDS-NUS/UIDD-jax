#!/bin/bash

nohup bash -c "CUDA_VISIBLE_DEVICES=1 python main_reduced_polymerF.py Model_name=HD2 model.seed=0" > l0.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python main_reduced_polymerF.py Model_name=HD2 model.seed=12" > l1.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python main_reduced_polymerF.py Model_name=HD2 model.seed=123" > l2.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=5 python main_reduced_polymerF.py Model_name=HD2 model.seed=1" > l3.file 2>&1 &
echo "Jobs are running in the background!" 