CUDA_VISIBLE_DEVICES=1 python main_sac.py --cuda  --version $1 --model_version $2 --target_version $3 --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_sac.py --cuda --version $1 --model_version $2 --target_version $3  --env Walker2d-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_sac.py --cuda --version $1 --model_version $2 --target_version $3  --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_sac.py --cuda --version $1 --model_version $2 --target_version $3  --env Hopper-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=2 python main_sac.py --cuda --version $1 --model_version $2 --target_version $3 --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_sac.py --cuda --version $1 --model_version $2 --target_version $3 --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=2 python main_sac.py --cuda --version $1 --model_version $2 --target_version $3  --env Ant-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_sac.py --cuda --version $1 --model_version $2 --target_version $3 --env Ant-v2 --seed 1


# CUDA_VISIBLE_DEVICES=3 python main_sac.py --cuda --version 0  --env Ant-v2 --seed 1 --start_steps 1000
# CUDA_VISIBLE_DEVICES=3 python main_sac.py --cuda --version 1  --env Ant-v2 --seed 1 --start_steps 1000
# CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy DVPG  --version 2  --tau 0.1  --env Hopper-v2 --seed 1