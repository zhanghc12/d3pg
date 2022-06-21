CUDA_VISIBLE_DEVICES=0 python offline_dppg_v2.py --env hopper-medium-v2 --seed 0 --alpha 0 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v2.py --env hopper-medium-expert-v2 --seed 0 --alpha 0 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v2.py --env hopper-expert-v2 --seed 0 --alpha 0 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=3 python offline_dppg_v2.py --env walker2d-medium-v2 --seed 0 --alpha 0 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=4 python offline_dppg_v2.py --env walker2d-medium-expert-v2 --seed 0 --alpha 0 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=5 python offline_dppg_v2.py --env walker2d-expert-v2 --seed 0 --alpha 0 --bc_scale $1 &\
CUDA_VISIBLE_DEVICES=5 python offline_dppg_v2.py --env halfcheetah-medium-v2 --seed 0 --alpha 0 --bc_scale $1




