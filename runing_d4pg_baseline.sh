CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy D4PG  --version $1  --target_threshold $2 --env Walker2d-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy D4PG  --version $1  --target_threshold $2  --env Walker2d-v2 --seed $4 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy D4PG  --version $1  --target_threshold $2  --env Hopper-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy D4PG  --version $1  --target_threshold $2  --env Hopper-v2 --seed $4 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy D4PG  --version $1  --target_threshold $2  --env HalfCheetah-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy D4PG  --version $1  --target_threshold $2  --env HalfCheetah-v2 --seed $4 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy D4PG  --version $1  --target_threshold $2  --env Ant-v2 --seed $3 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy D4PG  --version $1  --target_threshold $2  --env Ant-v2 --seed $4


# CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy D4PG  --version 6  --target_threshold 0.02  --env Ant-v2 --seed 1