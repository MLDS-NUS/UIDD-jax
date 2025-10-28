# !/bin/bash

nohup bash -c "CUDA_VISIBLE_DEVICES=0 python main_Kaczmarz_scale.py data.batch_size=1" > l1.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python main_Kaczmarz_scale.py data.batch_size=4" > l4.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python main_Kaczmarz_scale.py data.batch_size=16" > l16.file 2>&1 & 
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python main_Kaczmarz_scale.py data.batch_size=64" > l64.file 2>&1 & 


# nohup bash -c "CUDA_VISIBLE_DEVICES=0 python main_Kaczmarz_scale.py data.batch_size=2" > l2.file 2>&1 &
# nohup bash -c "CUDA_VISIBLE_DEVICES=1 python main_Kaczmarz_scale.py data.batch_size=8" > l8.file 2>&1 &
# nohup bash -c "CUDA_VISIBLE_DEVICES=2 python main_Kaczmarz_scale.py data.batch_size=32" > l32.file 2>&1 & 

echo "Jobs are running in the background!" 

