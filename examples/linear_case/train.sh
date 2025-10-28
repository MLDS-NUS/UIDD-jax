#!/bin/bash

nohup bash -c "CUDA_VISIBLE_DEVICES=1 python linear_case.py Model_name=HD2 model.seed=1 data.var=1" > l3.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python linear_case.py Model_name=HD2 model.seed=10 data.var=1" > l1.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python linear_case.py Model_name=HD2 model.seed=20 data.var=1" > l2.file 2>&1 &
 

nohup bash -c "CUDA_VISIBLE_DEVICES=1 python linear_case.py Model_name=HD2 model.seed=1 data.var=0.5" > 05l3.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python linear_case.py Model_name=HD2 model.seed=10 data.var=0.5" > 05l1.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python linear_case.py Model_name=HD2 model.seed=20 data.var=0.5" > 05l2.file 2>&1 &

nohup bash -c "CUDA_VISIBLE_DEVICES=1 python linear_case.py Model_name=HD2 model.seed=1 data.var=0.25" > 025l3.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python linear_case.py Model_name=HD2 model.seed=10 data.var=0.25" > 025l1.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python linear_case.py Model_name=HD2 model.seed=20 data.var=0.25" > 025l2.file 2>&1 &

nohup bash -c "CUDA_VISIBLE_DEVICES=1 python linear_case.py Model_name=HD2 model.seed=0 data.var=0" > 0l3.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python linear_case.py Model_name=HD2 model.seed=10 data.var=0" > 0l1.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python linear_case.py Model_name=HD2 model.seed=20 data.var=0" > 0l2.file 2>&1 &


nohup bash -c "CUDA_VISIBLE_DEVICES=1 python linear_case.py Model_name=HD2 model.seed=1 data.var=0.125" > 0125l3.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python linear_case.py Model_name=HD2 model.seed=10 data.var=0.125" > 0124l1.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python linear_case.py Model_name=HD2 model.seed=20 data.var=0.125" > 0124l2.file 2>&1 &

echo "Jobs are running in the background!" 