CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3 --target_threshold $4  --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3 --target_threshold $4  --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3 --target_threshold $4  --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3 --target_threshold $4  --env Ant-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3 --target_threshold $4  --env Walker2d-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3 --target_threshold $4  --env Hopper-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3 --target_threshold $4  --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3 --target_threshold $4  --env Ant-v2 --seed 1
