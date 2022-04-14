#CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2   --env Walker2d-v2 --seed 0 &\
#CUDA_VISIBLE_DEVICES=1 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2   --env Hopper-v2 --seed 0 &\
#CUDA_VISIBLE_DEVICES=2 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2  --env HalfCheetah-v2 --seed 0 &\
#CUDA_VISIBLE_DEVICES=3 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2  --env Ant-v2 --seed 0
#CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.1   --env Walker2d-v2 --seed 0

#CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.01  --env $1 --test 0 &\
#CUDA_VISIBLE_DEVICES=1 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.1   --env $1 --test 0 &\
#CUDA_VISIBLE_DEVICES=2 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 1.0   --env $1 --test 0 &\
#CUDA_VISIBLE_DEVICES=3 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.0   --env $1 --test 0

CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 1  --target_threshold 0.0001  --env $1 --test 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dnpg.py --policy D4PG  --version 1  --target_threshold 0.001   --env $1 --test 0 &\
CUDA_VISIBLE_DEVICES=2 python main_dnpg.py --policy D4PG  --version 1  --target_threshold 0.01   --env $1 --test 0 &\
CUDA_VISIBLE_DEVICES=3 python main_dnpg.py --policy D4PG  --version 1  --target_threshold 0.1   --env $1 --test 0


#sh eval_runing_dnpg_normal_4gpu.sh Walker2d-v2
#sh eval_runing_dnpg_normal_4gpu.sh Hopper-v2
#sh eval_runing_dnpg_normal_4gpu.sh HalfCheetah-v2
#sh eval_runing_dnpg_normal_4gpu.sh Ant-v2





