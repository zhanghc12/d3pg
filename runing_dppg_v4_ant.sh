CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env antmaze-umaze-v0 --seed 0 --alpha 2 --bc_scale 0.1 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env antmaze-umaze-diverse-v0 --seed 0 --alpha 2 --bc_scale 0.1 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env antmaze-medium-diverse-v0 --seed 0 --alpha 2 --bc_scale 0.1 &\
CUDA_VISIBLE_DEVICES=3 python offline_dppg_v4.py --env antmaze-medium-play-v0 --seed 0 --alpha 2 --bc_scale 0.1 &\
CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env antmaze-large-diverse-v0 --seed 0 --alpha 2 --bc_scale 0.1 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env antmaze-large-play-v0 --seed 0 --alpha 2 --bc_scale 0.1 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env antmaze-umaze-v0 --seed 0 --alpha 2 --bc_scale 1 &\
CUDA_VISIBLE_DEVICES=3 python offline_dppg_v4.py --env antmaze-umaze-diverse-v0 --seed 0 --alpha 2 --bc_scale 1 &\
CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env antmaze-medium-diverse-v0 --seed 0 --alpha 2 --bc_scale 1 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env antmaze-medium-play-v0 --seed 0 --alpha 2 --bc_scale 1 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env antmaze-large-diverse-v0 --seed 0 --alpha 2 --bc_scale 1 &\
CUDA_VISIBLE_DEVICES=3 python offline_dppg_v4.py --env antmaze-large-play-v0 --seed 0 --alpha 2 --bc_scale 1 &\
CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env antmaze-umaze-v0 --seed 0 --alpha 2 --bc_scale 10 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env antmaze-umaze-diverse-v0 --seed 0 --alpha 2 --bc_scale 10 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env antmaze-medium-diverse-v0 --seed 0 --alpha 2 --bc_scale 10 &\
CUDA_VISIBLE_DEVICES=3 python offline_dppg_v4.py --env antmaze-medium-play-v0 --seed 0 --alpha 2 --bc_scale 10 &\
CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env antmaze-large-diverse-v0 --seed 0 --alpha 2 --bc_scale 10 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env antmaze-large-play-v0 --seed 0 --alpha 2 --bc_scale 10 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env antmaze-umaze-v0 --seed 0 --alpha 2 --bc_scale 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_dppg_v4.py --env antmaze-umaze-diverse-v0 --seed 0 --alpha 2 --bc_scale 0 &\
CUDA_VISIBLE_DEVICES=0 python offline_dppg_v4.py --env antmaze-medium-diverse-v0 --seed 0 --alpha 2 --bc_scale 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_dppg_v4.py --env antmaze-medium-play-v0 --seed 0 --alpha 2 --bc_scale 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_dppg_v4.py --env antmaze-large-diverse-v0 --seed 0 --alpha 2 --bc_scale 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_dppg_v4.py --env antmaze-large-play-v0 --seed 0 --alpha 2 --bc_scale 0

# CUDA_VISIBLE_DEVICES=1 python offline_dppg_v3.py --env hopper-medium-replay-v2 --seed 0 --alpha 0 --bc_scale 0.1

