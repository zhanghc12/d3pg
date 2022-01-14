CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2 --env hopper-random-v0 --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env hopper-medium-v0  --seed $3 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env hopper-medium-expert-v0  --seed $3 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env hopper-medium-replay-v0 --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env hopper-expert-v0 --seed $3 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2 --env halfcheetah-random-v0 --seed $3 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env halfcheetah-medium-v0  --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env halfcheetah-medium-expert-v0  --seed $3 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env halfcheetah-medium-replay-v0 --seed $3 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env halfcheetah-expert-v0 --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2 --env walker2d-random-v0 --seed $3 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env walker2d-medium-v0  --seed $3 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env walker2d-medium-expert-v0  --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env walker2d-medium-replay-v0 --seed $3 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env walker2d-expert-v0 --seed $3


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05

# CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 20 --env hopper-random-v0 --seed 0



