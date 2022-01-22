CUDA_VISIBLE_DEVICES=4 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2 --env pen-human-v0 --seed $3 --top_quantiles_to_drop $4 &\
CUDA_VISIBLE_DEVICES=5 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2  --env hammer-human-v0  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=6 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2  --env door-human-v0  --seed $3 --top_quantiles_to_drop $4  &\
CUDA_VISIBLE_DEVICES=7 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2  --env relocate-human-v0 --seed $3 --top_quantiles_to_drop $4  &\

#CUDA_VISIBLE_DEVICES=0 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2 --env pen-human-v0 --seed $3 --top_quantiles_to_drop $4 &\
#CUDA_VISIBLE_DEVICES=1 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2  --env hammer-human-v0  --seed $3 --top_quantiles_to_drop $4  &\
#CUDA_VISIBLE_DEVICES=2 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2  --env door-human-v0  --seed $3 --top_quantiles_to_drop $4  &\
#CUDA_VISIBLE_DEVICES=3 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2  --env relocate-human-v0 --seed $3 --top_quantiles_to_drop $4  &\
#CUDA_VISIBLE_DEVICES=4 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2  --env pen-cloned-v0 --seed $3 --top_quantiles_to_drop $4  &\
#CUDA_VISIBLE_DEVICES=5 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2 --env hammer-cloned-v0 --seed $3 --top_quantiles_to_drop $4  &\
#CUDA_VISIBLE_DEVICES=6 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2  --env door-cloned-v0  --seed $3 --top_quantiles_to_drop $4  &\
#CUDA_VISIBLE_DEVICES=7 python offline_kd_final_adroit.py --policy TQC  --version $1  --bc_scale $2  --env relocate-cloned-v0  --seed $3 --top_quantiles_to_drop $4

# CUDA_VISIBLE_DEVICES=6 python offline_kd_final.py --policy TQC  --version 3  --bc_scale 20  --env walker2d-expert-v2 --seed 0 --top_quantiles_to_drop 0


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05


# sh icml_final.sh 1 20 0 0
