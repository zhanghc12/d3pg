CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 150  --n_nets 10 --env halfcheetah-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 150  --n_nets 10 --env halfcheetah-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 150  --n_nets 10  --env walker2d-medium-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 175  --n_nets 10 --env halfcheetah-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 175  --n_nets 10 --env halfcheetah-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 175  --n_nets 10  --env walker2d-medium-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 125  --n_nets 10 --env halfcheetah-medium-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 125  --n_nets 10 --env halfcheetah-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python offline_tqc.py --policy TQC  --version 0  --top_quantiles_to_drop_per_net 125  --n_nets 10  --env walker2d-medium-v2  --seed 0



# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



