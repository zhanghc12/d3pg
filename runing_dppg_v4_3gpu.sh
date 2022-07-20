CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env hopper-medium-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env hopper-medium-replay-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env hopper-medium-expert-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env hopper-random-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env walker2d-medium-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env walker2d-medium-replay-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env walker2d-medium-expert-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env walker2d-random-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env halfcheetah-medium-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env halfcheetah-medium-replay-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env halfcheetah-medium-expert-v2 --seed 0 --alpha $2 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env halfcheetah-random-v2 --seed 0 --alpha $2 --bc_scale $1

# CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env halfcheetah-medium-expert-v2 --seed 0 --alpha 0 --bc_scale $1


# CUDA_VISIBLE_DEVICES=1 python offline_dppg_v3.py --env hopper-medium-replay-v2 --seed 0 --alpha 0 --bc_scale 0.1

