CUDA_VISIBLE_DEVICES=0 python main_mbpo_eval.py --version 12 --seed 10 --env_name Walker2d-v2 &\
CUDA_VISIBLE_DEVICES=1 python main_mbpo_eval.py --version 12 --seed 11 --env_name Walker2d-v2 &\
CUDA_VISIBLE_DEVICES=2 python main_mbpo_eval.py --version 12 --seed 12 --env_name Walker2d-v2 &\
CUDA_VISIBLE_DEVICES=3 python main_mbpo_eval.py --version 12 --seed 13 --env_name Walker2d-v2 &\
CUDA_VISIBLE_DEVICES=4 python main_mbpo_eval.py --version 12 --seed 10 --model_type tensorflow --env_name Walker2d-v2 &\
CUDA_VISIBLE_DEVICES=5 python main_mbpo_eval.py --version 12 --seed 11 --model_type tensorflow --env_name Walker2d-v2 &\
CUDA_VISIBLE_DEVICES=6 python main_mbpo_eval.py --version 12 --seed 12 --model_type tensorflow --env_name Walker2d-v2 &\
CUDA_VISIBLE_DEVICES=7 python main_mbpo_eval.py --version 12 --seed 13 --model_type tensorflow --env_name Walker2d-v2



#CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --version 3 --seed 0 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --version 4 --seed 0 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 0 --seed 1 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 3 --seed 1 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 4 --seed 1 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=2 python main_mbpo.py --version 0 --seed 0 --env_name Hopper-v2 &\
#CUDA_VISIBLE_DEVICES=2 python main_mbpo.py --version 3 --seed 0 --env_name Hopper-v2 &\
#CUDA_VISIBLE_DEVICES=2 python main_mbpo.py --version 4 --seed 0 --env_name Hopper-v2 &\
#CUDA_VISIBLE_DEVICES=3 python main_mbpo.py --version 0 --seed 1 --env_name Hopper-v2 &\
#CUDA_VISIBLE_DEVICES=3 python main_mbpo.py --version 3 --seed 1 --env_name Hopper-v2 &\
#CUDA_VISIBLE_DEVICES=3 python main_mbpo.py --version 4 --seed 1 --env_name Hopper-v2

#CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 1 --seed 0 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 2 --seed 0 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 4 --seed 0 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 1 --seed 1 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 2 --seed 1 --env_name Walker2d-v2 &\
#CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 4 --seed 1 --env_name Walker2d-v2

# CUDA_VISIBLE_DEVICES=1 python main_mbpo.py --version 3 --seed 1

# CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --version 1 --seed 3 --env_name InvertedPendulum-v0
# CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --version 2 --seed 3 --env_name InvertedPendulum-v0
# CUDA_VISIBLE_DEVICES=0 python main_mbpo.py --version 4 --seed 3 --env_name InvertedPendulum-v0
