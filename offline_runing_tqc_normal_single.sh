CUDA_VISIBLE_DEVICES=7 python offline_tqc.py --policy TQC  --version $1  --top_quantiles_to_drop_per_net $2 --env hopper-medium-replay-v0 --seed 0


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



