#!/bin/bash


nohup bash -c "CUDA_VISIBLE_DEVICES=2 python main_reduced_polymerF.py Model_name=HD2 model.seed=0" > l0.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python main_reduced_polymerF.py Model_name=HD2 model.seed=1" > l1.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python main_reduced_polymerF.py Model_name=HD2 model.seed=12" > l12.file 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python main_reduced_polymerF.py Model_name=HD2 model.seed=123" > l123.file 2>&1 &

# nohup bash -c "CUDA_VISIBLE_DEVICES=4 python main_reduced_polymerF_test.py Model_name=HD2 model.seed=0" > l10.file 2>&1 &
# nohup bash -c "CUDA_VISIBLE_DEVICES=2 python main_reduced_polymerF_test.py Model_name=HD2 model.seed=12 Model_diff=constant" > l1.file 2>&1 &
# nohup bash -c "CUDA_VISIBLE_DEVICES=3 python main_reduced_polymerF_test.py Model_name=HD2 model.seed=0 Model_diff=constant" > l2.file 2>&1 &

# nohup bash -c "CUDA_VISIBLE_DEVICES=4 python main_reduced_polymerF_test.py Model_name=HD2 model.seed=12 Model_diff=diagMLP" > l3.file 2>&1 &
# nohup bash -c "CUDA_VISIBLE_DEVICES=5 python main_reduced_polymerF_test.py Model_name=HD2 model.seed=0 Model_diff=diagMLP" > l4.file 2>&1 &

# nohup bash -c "CUDA_VISIBLE_DEVICES=6 python main_reduced_polymerF_test.py Model_name=HD2 model.seed=12 Model_diff=MLP" > l5.file 2>&1 &
# nohup bash -c "CUDA_VISIBLE_DEVICES=7 python main_reduced_polymerF_test.py Model_name=HD2 model.seed=0 Model_diff=MLP" > l6.file 2>&1 &




# nohup bash -c "CUDA_VISIBLE_DEVICES=3 python main_reduced_polymerF_test.py Model_name=HD2 model.seed=123" > l2.file 2>&1 &
# nohup bash -c "CUDA_VISIBLE_DEVICES=3 python main_reduced_polymerF_test.py Model_name=HD2 model.seed=0 Model_diff=constant" > l3.file 2>&1 &
echo "Jobs are running in the background!" 