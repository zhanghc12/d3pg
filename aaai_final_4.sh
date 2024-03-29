CUDA_VISIBLE_DEVICES=0 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2 --env hopper-random-v2 --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env hopper-medium-v2   --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env hopper-medium-expert-v2   --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env hopper-medium-replay-v2 --k $3  --is_random $4 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env hopper-expert-v2 --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2 --env halfcheetah-random-v2 --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env halfcheetah-medium-v2  --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env halfcheetah-medium-expert-v2  --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env halfcheetah-medium-replay-v2 --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env halfcheetah-expert-v2 --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2 --env walker2d-random-v2 --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env walker2d-medium-v2  --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env walker2d-medium-expert-v2  --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env walker2d-medium-replay-v2 --k $3 --is_random $4 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_aaai_vae.py --policy TQC  --version $1  --eta $2  --env walker2d-expert-v2 --k $3 --is_random $4


# CUDA_VISIBLE_DEVICES=0 python offline_kd_aaai_vae.py --policy TQC  --version 9  --eta 200  --env walker2d-medium-v2 --seed 0 --top_quantiles_to_drop 0 --k 2 --is_random $4
# CUDA_VISIBLE_DEVICES=1 python offline_kd_aaai_vae.py --policy TQC  --version 9  --eta 200  --env halfcheetah-medium-replay-v2 --seed 0 --top_quantiles_to_drop 0 --k 1 --is_random $4

# CUDA_VISIBLE_DEVICES=6 python offline_kd_aaai_vae.py --policy TQC  --version 3  --eta 20  --env walker2d-expert-v2 --seed 0 --top_quantiles_to_drop 0


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.0512121211


# sh icml_final.sh 1 20 0 0
