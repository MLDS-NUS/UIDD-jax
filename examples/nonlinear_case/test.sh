CUDA_VISIBLE_DEVICES=7 python evaluate_mmd.py --dt 0.01 --kappa 0.8 \
    --seeds 0 1 12 123 1234 \
    --models OnsagerRegL21e-3 OnsagerRegL21e-4 OnsagerRegSobolev1e-3 OnsagerRegSobolev1e-4 \
    --out-name mmd_summary_OnsagerReg_dt0.01