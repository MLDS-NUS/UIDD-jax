# !/bin/bash

nohup bash -c "CUDA_VISIBLE_DEVICES=0 python main_sgld.py model.seed=10" > s10.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python main_sgld.py model.seed=20" > s20.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python main_sgld.py model.seed=30" > s30.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python main_sgld.py model.seed=40" > s40.file 2>&1 &

echo "Jobs are running in the background!" 

