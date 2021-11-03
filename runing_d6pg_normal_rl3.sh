CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy D6PG  --version $1  --target_threshold $2 --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy D6PG  --version $1  --target_threshold $2  --env Walker2d-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy D6PG  --version $1  --target_threshold $2  --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy D6PG  --version $1  --target_threshold $2  --env Hopper-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy D6PG  --version $1  --target_threshold $2  --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy D6PG  --version $1  --target_threshold $2  --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy D6PG  --version $1  --target_threshold $2  --env Ant-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy D6PG  --version $1  --target_threshold $2  --env Ant-v2 --seed 1