CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 5  --env halfcheetah-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 20  --env halfcheetah-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 50  --env halfcheetah-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 50  --env hopper-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=4 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 300  --env hopper-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=5 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 100  --env hopper-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=6 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 50  --env walker2d-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=7 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 300  --env walker2d-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 100  --env walker2d-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 3.25 --env hopper-medium-v2  --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 3.75  --env hopper-medium-v2  --seed $1 --output_dim 9 &\

# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05

