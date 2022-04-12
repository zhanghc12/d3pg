CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2   --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2   --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2  --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2  --env Ant-v2 --seed 0



CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.1   --env Walker2d-v2 --seed 0

CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.001   --env InvertedPendulum-v2 --test 0
CUDA_VISIBLE_DEVICES=1 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.   --env InvertedPendulum-v2 --test 0
CUDA_VISIBLE_DEVICES=2 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.01   --env InvertedPendulum-v2 --test 0


