CUDA_VISIBLE_DEVICES=0 python svgp_cloning.py --env hopper-medium-v2  --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python svgp_cloning.py --env hopper-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python svgp_cloning.py --env hopper-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python svgp_cloning.py --env walker2d-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=4 python svgp_cloning.py --env walker2d-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=5 python svgp_cloning.py --env hopper-random-v2  --seed 1 &\
CUDA_VISIBLE_DEVICES=6 python svgp_cloning.py --env halfcheetah-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=7 python svgp_cloning.py --env halfcheetah-expert-v2 --seed 0

# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



