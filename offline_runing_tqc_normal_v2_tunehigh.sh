CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 223  --n_nets 10 --env hopper-medium-v2  --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 223  --n_nets 10 --env hopper-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 233  --n_nets 10 --env hopper-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 223 --n_nets  10  --env walker2d-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=4 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 223 --n_nets  10  --env walker2d-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=5 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 225  --n_nets 10 --env hopper-medium-v2  --seed 1 &\
CUDA_VISIBLE_DEVICES=6 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 225  --n_nets 10 --env hopper-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=7 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 235  --n_nets 10 --env hopper-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 225 --n_nets  10  --env walker2d-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 225 --n_nets  10  --env walker2d-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 227  --n_nets 10 --env hopper-medium-v2  --seed 1 &\
CUDA_VISIBLE_DEVICES=3 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 227  --n_nets 10 --env hopper-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=4 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 237  --n_nets 10 --env hopper-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=5 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 227 --n_nets  10  --env walker2d-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=6 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 227 --n_nets  10  --env walker2d-expert-v2 --seed 0


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



