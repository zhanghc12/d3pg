CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --model_type pytorch --version $1 --seed 0 --env_name Walker2d-v2 &\
CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --model_type pytorch --version $1 --seed 1 --env_name Walker2d-v2


