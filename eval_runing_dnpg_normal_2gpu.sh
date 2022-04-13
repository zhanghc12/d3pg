CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.0001  --env $1 --test 1 &\
CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.001   --env $1 --test 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.01   --env $1 --test 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.1   --env $1 --test 1
