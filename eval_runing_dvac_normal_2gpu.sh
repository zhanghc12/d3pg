CUDA_VISIBLE_DEVICES=0 python main_dueling_eval.py --policy Dueling  --version $1  --target_threshold $2 --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling_eval.py --policy Dueling  --version $1  --target_threshold $2  --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling_eval.py --policy Dueling  --version $1  --target_threshold $2  --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling_eval.py --policy Dueling  --version $1  --target_threshold $2  --env Ant-v2 --seed 0
