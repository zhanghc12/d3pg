CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 200 --env hopper-random-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 4  --env hopper-medium-v2  --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 20  --env hopper-medium-expert-v2  --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 200  --env hopper-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=4 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 200  --env hopper-expert-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=5 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 200 --env halfcheetah-random-v2 --seed $1 --output_dim 6 &\
CUDA_VISIBLE_DEVICES=6 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 200  --env halfcheetah-medium-v2  --seed $1 --output_dim 6 &\
CUDA_VISIBLE_DEVICES=7 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 20  --env halfcheetah-medium-expert-v2  --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 10  --env halfcheetah-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 20  --env halfcheetah-expert-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 200 --env walker2d-random-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=4 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 15  --env walker2d-medium-v2  --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=5 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 3.25  --env walker2d-medium-expert-v2  --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=6 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 200  --env walker2d-medium-replay-v2 --seed $1 --output_dim 9 &\
CUDA_VISIBLE_DEVICES=7 python offline_kd_final.py --policy TQC  --version 1  --bc_scale 3.25  --env walker2d-expert-v2 --seed $1 --output_dim 9


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05




halfcheetah-random 3
halfcheetah-medium
halfcheetah-medium-replay
halfcheetah-medium-expert
halfcheetahr-expert

hopper-random
hopper-medium
hopper-medium-replay
hopper-medium-expert
hopper-expert

walker2d-random
walker2d-medium
walker2d-medium-replay
walker2d-medium-expert
walker2d-expert