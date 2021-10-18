CUDA_VISIBLE_DEVICES=4 python main_dueling.py --policy DHPG  --env HalfCheetah-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=5 python main_dueling.py --policy DHPG  --env HalfCheetah-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=6 python main_dueling.py --policy DHPG  --env Hopper-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=7 python main_dueling.py --policy DHPG  --env Hopper-v3 --seed 1