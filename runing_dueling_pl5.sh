CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 3 --env HalfCheetah-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling   --version 3 --env HalfCheetah-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 4  --env HalfCheetah-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 4  --env HalfCheetah-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 3  --env Hopper-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling   --version 3 --env Hopper-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 4  --env Hopper-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 4  --env Hopper-v3 --seed 1