CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3  --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3  --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3  --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --exp_num_critic $3  --env Ant-v2 --seed 0
