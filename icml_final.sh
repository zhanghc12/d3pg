CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2 --env hopper-random-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env hopper-medium-v2  --seed $3 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env hopper-medium-expert-v2  --seed $3 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env hopper-medium-replay-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env hopper-expert-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2 --env halfcheetah-random-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env halfcheetah-medium-v2  --seed $3 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env halfcheetah-medium-expert-v2  --seed $3 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env halfcheetah-medium-replay-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env halfcheetah-expert-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2 --env walker2d-random-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env walker2d-medium-v2  --seed $3 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env walker2d-medium-expert-v2  --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env walker2d-medium-replay-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final.py --policy TQC  --version $1  --bc_scale $2  --env walker2d-expert-v2 --seed $3


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



