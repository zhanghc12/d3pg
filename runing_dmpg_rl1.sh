CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy DMPG  --version 2 --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy DMPG  --version 2 --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy DMPG  --version 2 --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy DMPG  --version 2 --env Walker2d-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=4 python main_dueling.py --policy DMPG  --version 0 --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=5 python main_dueling.py --policy DMPG  --version 0 --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=6 python main_dueling.py --policy DMPG  --version 0 --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=7 python main_dueling.py --policy DMPG  --version 0 --env Walker2d-v2 --seed 1

