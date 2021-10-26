CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy OurDDPG  --version 0 --env Walker2d-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy OurDDPG  --version 0 --env Walker2d-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 2  --env Walker2d-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 2  --env Walker2d-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 2  --env HalfCheetah-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 2  --env HalfCheetah-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 2  --env Hopper-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 2  --env Hopper-v3 --seed 1