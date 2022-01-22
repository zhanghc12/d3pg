CUDA_VISIBLE_DEVICES=0 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 200 --env pen-human-v0 --seed $3 --top_quantiles_to_drop $4 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 200  --env pen-cloned-v0  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 200  --env door-human-v0  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 200  --env door-cloned-v0 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 150 --env pen-human-v0 --seed $3 --top_quantiles_to_drop $4 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 150  --env pen-cloned-v0  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 150  --env door-human-v0  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 150  --env door-cloned-v0 --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 250 --env pen-human-v0 --seed $3 --top_quantiles_to_drop $4 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 250  --env pen-cloned-v0  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=1 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 250  --env door-human-v0  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=2 python offline_kd_final_adroit.py --policy TQC  --version 1  --bc_scale 250  --env door-cloned-v0 --seed $3 --top_quantiles_to_drop $4


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05

# CUDA_VISIBLE_DEVICES=0 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 20 --env hopper-random-v2 --seed 0



