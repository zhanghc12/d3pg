CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --env HalfCheetah-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --env HalfCheetah-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy OurDDPG  --env HalfCheetah-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy OurDDPG  --env HalfCheetah-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy Dueling  --env Hopper-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy Dueling  --env Hopper-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy OurDDPG  --env Hopper-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy OurDDPG  --env Hopper-v3 --seed 1