CUDA_VISIBLE_DEVICES=0 python main_dnpg_v4.py --policy D4PG  --version $1 --ratio $2  --target_threshold 1 --seed 11 --env  $3 &\
CUDA_VISIBLE_DEVICES=1 python main_dnpg_v4.py --policy D4PG  --version $1 --ratio $2  --target_threshold 0.1 --seed 12  --env $3  &\
CUDA_VISIBLE_DEVICES=2 python main_dnpg_v4.py --policy D4PG  --version $1 --ratio $2  --target_threshold 0.01 --seed 13  --env $3   &\
CUDA_VISIBLE_DEVICES=3 python main_dnpg_v4.py --policy D4PG  --version $1 --ratio $2  --target_threshold 0 --seed 14  --env $3


#   CUDA_VISIBLE_DEVICES=3 python main_dnpg_v4.py --policy D4PG  --version 0 --ratio 0  --target_threshold 0 --seed 14  --env HalfCheetah-v2