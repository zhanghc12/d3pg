CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 0 --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling   --version 0 --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 2  --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 2  --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 0  --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling   --version 0 --env Hopper-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 2  --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 2  --env Hopper-v2 --seed 1