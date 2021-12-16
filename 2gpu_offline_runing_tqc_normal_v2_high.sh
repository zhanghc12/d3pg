CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --policy TQC  --version $1  --top_quantiles_to_drop_per_net $2  --n_nets $3 --bc_scale $4 --env hopper-medium-v2  --seed 1 &\
CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --policy TQC  --version $1  --top_quantiles_to_drop_per_net $2  --n_nets $3 --bc_scale $4 --env hopper-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version $1  --top_quantiles_to_drop_per_net $2  --n_nets $3 --bc_scale $4 --env hopper-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version $1  --top_quantiles_to_drop_per_net $2  --n_nets $3 --bc_scale $4 --env walker2d-expert-v2 --seed 0


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



