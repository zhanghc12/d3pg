CUDA_VISIBLE_DEVICES=0 python main_exp.py --policy Dueling  --version $1 --exp_version $2 --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Ant-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Walker2d-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Hopper-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling   --version $1 --exp_version $2  --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Ant-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=0 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Walker2d-v2 --seed 2 &\
CUDA_VISIBLE_DEVICES=1 python main_exp.py --policy Dueling   --version $1 --exp_version $2  --env Hopper-v2 --seed 2 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling   --version $1 --exp_version $2  --env HalfCheetah-v2 --seed 2 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling   --version $1 --exp_version $2  --env Ant-v2 --seed 2 &\
CUDA_VISIBLE_DEVICES=0 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Walker2d-v2 --seed 3 &\
CUDA_VISIBLE_DEVICES=1 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Hopper-v2 --seed 3 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env HalfCheetah-v2 --seed 3 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling   --version $1 --exp_version $2  --env Ant-v2 --seed 3 &\
CUDA_VISIBLE_DEVICES=0 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Walker2d-v2 --seed 4 &\
CUDA_VISIBLE_DEVICES=1 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Hopper-v2 --seed 4 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling --version $1 --exp_version $2  --env HalfCheetah-v2 --seed 4 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling   --version $1 --exp_version $2 --env Ant-v2 --seed 4 &\
CUDA_VISIBLE_DEVICES=0 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Walker2d-v2 --seed 5 &\
CUDA_VISIBLE_DEVICES=1 python main_exp.py --policy Dueling   --version $1 --exp_version $2  --env Hopper-v2 --seed 5 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env HalfCheetah-v2 --seed 5 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling   --version $1 --exp_version $2  --env Ant-v2 --seed 5 &\
CUDA_VISIBLE_DEVICES=0 python main_exp.py --policy Dueling   --version $1 --exp_version $2  --env Walker2d-v2 --seed 6 &\
CUDA_VISIBLE_DEVICES=1 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Hopper-v2 --seed 6 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling   --version $1 --exp_version $2  --env HalfCheetah-v2 --seed 6 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Ant-v2 --seed 6 &\
CUDA_VISIBLE_DEVICES=0 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Walker2d-v2 --seed 7 &\
CUDA_VISIBLE_DEVICES=1 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env Hopper-v2 --seed 7 &\
CUDA_VISIBLE_DEVICES=2 python main_exp.py --policy Dueling  --version $1 --exp_version $2  --env HalfCheetah-v2 --seed 7 &\
CUDA_VISIBLE_DEVICES=3 python main_exp.py --policy Dueling   --version $1 --exp_version $2 --env Ant-v2 --seed 7
