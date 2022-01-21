CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 200  --env halfcheetah-expert-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 300  --env halfcheetah-expert-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 12.5  --env walker2d-medium-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 3.25  --env walker2d-medium-expert-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 3.75  --env walker2d-medium-expert-v2  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 3.25  --env walker2d-expert-v2 --seed $3 --top_quantiles_to_drop $4 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 3.75  --env walker2d-expert-v2 --seed $3 --top_quantiles_to_drop $4 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 0.125 --env halfcheetah-random-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 0.0625 --env halfcheetah-random-v2 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale 17.5  --env walker2d-medium-v2  --seed $3 --top_quantiles_to_drop $4


# CUDA_VISIBLE_DEVICES=6 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 20  --env walker2d-expert-v2 --seed 0 --top_quantiles_to_drop 0


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05


# sh icml_final.sh 1 20 0 0
