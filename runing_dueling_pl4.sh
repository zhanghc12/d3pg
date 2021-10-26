CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 0 --env Walker2d-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling   --version 0 --env Walker2d-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 1  --env Walker2d-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy Dueling  --version 1  --env Walker2d-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 0  --env Humanoid-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling   --version 0 --env Humanoid-v3 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 1  --env Humanoid-v3 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy Dueling  --version 1  --env Humanoid-v3 --seed 1