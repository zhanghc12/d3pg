#CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2   --env Walker2d-v2 --seed 0 &\
#CUDA_VISIBLE_DEVICES=1 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2   --env Hopper-v2 --seed 0 &\
#CUDA_VISIBLE_DEVICES=2 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2  --env HalfCheetah-v2 --seed 0 &\
#CUDA_VISIBLE_DEVICES=3 python main_dnpg.py --policy D4PG  --version $1  --target_threshold $2  --env Ant-v2 --seed 0
#CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.1   --env Walker2d-v2 --seed 0

#CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.01  --env $1 --test 0 &\
#CUDA_VISIBLE_DEVICES=1 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.1   --env $1 --test 0 &\
#CUDA_VISIBLE_DEVICES=2 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 1.0   --env $1 --test 0 &\
#CUDA_VISIBLE_DEVICES=3 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.0   --env $1 --test 0

CUDA_VISIBLE_DEVICES=0 python main_dnpg_v3.py --policy D4PG  --version $2  --target_threshold 0.0001  --env $1  &\
CUDA_VISIBLE_DEVICES=1 python main_dnpg_v3.py --policy D4PG  --version $2  --target_threshold 0.001   --env $1  &\
CUDA_VISIBLE_DEVICES=2 python main_dnpg_v3.py --policy D4PG  --version $2  --target_threshold 0.01   --env $1   &\
CUDA_VISIBLE_DEVICES=3 python main_dnpg_v3.py --policy D4PG  --version $2  --target_threshold 0.1   --env $1


#sh eval_ac_dnpg_4gpu.sh Walker2d-v2 3
#sh eval_ac_dnpg_4gpu.sh Hopper-v2 3
#sh eval_ac_dnpg_4gpu.sh HalfCheetah-v2 3
#sh eval_ac_dnpg_4gpu.sh Ant-v2 3


# CUDA_VISIBLE_DEVICES=0 python main_dnpg.py --policy D4PG  --version 0  --target_threshold 0.0001  --env HalfCheetah-v2 --test 0

# CUDA_VISIBLE_DEVICES=0 python main_dnpg_v2.py --policy D4PG  --version 4  --target_threshold 0.0001  --env Hopper-v2 --test 1 --first_phase 1




