CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy DHPG  --version 0 --env HalfCheetah-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy DHPG  --version 0 --env Hopper-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy DHPG  --version 1 --env HalfCheetah-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy DHPG  --version 1 --env Hopper-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=4 python main_dueling.py --policy DHPG  --version 2 --env HalfCheetah-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=5 python main_dueling.py --policy DHPG  --version 2 --env Hopper-v3 --seed 0