CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy OurDDPG  --version 0 --env Walker2d-v1 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy OurDDPG  --version 0 --env Walker2d-v1 --seed 1 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy OurDDPG  --version 0  --env Hopper-v1 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy OurDDPG  --version 0  --env Hopper-v1 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy OurDDPG  --version 0  --env HalfCheetah-v1 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy OurDDPG  --version 0  --env HalfCheetah-v1 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy OurDDPG  --version 0  --env Ant-v1 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy OurDDPG  --version 0  --env Ant-v1 --seed 1