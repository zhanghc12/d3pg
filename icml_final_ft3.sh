CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 7 --env hopper-random-v2 --seed $3 --top_quantiles_to_drop $4 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 4.5  --env hopper-medium-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 3.5  --env hopper-medium-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 30  --env hopper-medium-expert-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=4 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 40  --env hopper-medium-expert-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=5 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 6  --env hopper-medium-replay-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=6 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 7  --env hopper-medium-replay-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=7 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 200  --env hopper-expert-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 300  --env hopper-expert-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 0.25 --env halfcheetah-random-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 0.5  --env halfcheetah-medium-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 0.25 --env halfcheetah-medium-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=4 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 30  --env halfcheetah-medium-expert-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=5 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 40  --env halfcheetah-medium-expert-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=6 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 1.6  --env halfcheetah-medium-replay-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=7 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 1.8  --env halfcheetah-medium-replay-v2 --seed $3 --top_quantiles_to_drop $4

# CUDA_VISIBLE_DEVICES=6 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 20  --env walker2d-expert-v2 --seed 0 --top_quantiles_to_drop 0


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05


# sh icml_final.sh 1 20 0 0
