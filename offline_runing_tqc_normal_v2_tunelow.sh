CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 130  --n_nets 10 --env halfcheetah-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 135  --n_nets 10 --env halfcheetah-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 140  --n_nets 10 --env halfcheetah-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 145  --n_nets 10 --env halfcheetah-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 180  --n_nets 10 --env halfcheetah-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 185  --n_nets 10 --env halfcheetah-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 190  --n_nets 10 --env halfcheetah-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 195  --n_nets 10 --env halfcheetah-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 130  --n_nets 10  --env walker2d-medium-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 140  --n_nets 10  --env walker2d-medium-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 150  --n_nets 10  --env walker2d-medium-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 160  --n_nets 10  --env walker2d-medium-v2  --seed 0


# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



