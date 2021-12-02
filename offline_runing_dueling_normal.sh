CUDA_VISIBLE_DEVICES=0 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2 --env hopper-random-v0 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env hopper-medium-v0  --seed 1 &\
CUDA_VISIBLE_DEVICES=2 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env hopper-medium-expert-v0  --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env hopper-medium-replay-v0 --seed 1 &\
CUDA_VISIBLE_DEVICES=4 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env hopper-expert-v0 --seed 0 &\
CUDA_VISIBLE_DEVICES=5 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2 --env halfcheetah-random-v0 --seed 0 &\
CUDA_VISIBLE_DEVICES=6 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env halfcheetah-medium-v0  --seed 1 &\
CUDA_VISIBLE_DEVICES=7 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env halfcheetah-medium-expert-v0  --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env halfcheetah-medium-replay-v0 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env halfcheetah-expert-v0 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2 --env walker2d-random-v0 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env walker2d-medium-v0  --seed 1 &\
CUDA_VISIBLE_DEVICES=4 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env walker2d-medium-expert-v0  --seed 0 &\
CUDA_VISIBLE_DEVICES=5 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env walker2d-medium-replay-v0 --seed 1 &\
CUDA_VISIBLE_DEVICES=6 python offline_dueling.py --policy Dueling  --version $1  --target_threshold $2  --env walker2d-expert-v0 --seed 0


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



