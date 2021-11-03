CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy D2PG --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy D2PG --env Walker2d-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy D2PG --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy D2PG --env Hopper-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy D2PG --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy D2PG --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy D2PG --env Ant-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy D2PG --env Ant-v2 --seed 1