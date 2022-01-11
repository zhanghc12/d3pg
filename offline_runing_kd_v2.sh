CUDA_VISIBLE_DEVICES=0 python offline_kd.py --env hopper-expert-v2  --bc_scale 0 --seed 1 --loading 0 --k 10 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd.py --env hopper-random-v2  --bc_scale 0 --seed 1 --loading 0 --k 10 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd.py --env hopper-medium-v2  --bc_scale 0 --seed 1 --loading 0 --k 10 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd.py --env hopper-medium-exert-v2  --bc_scale 0 --seed 1 --loading 0 --k 10 &\
CUDA_VISIBLE_DEVICES=0 python offline_kd.py --env hopper-medium-replay-v2  --bc_scale 0 --seed 1 --loading 0 --k 10 &\
CUDA_VISIBLE_DEVICES=1 python offline_kd.py --env walker2d-expert-v2  --bc_scale 0 --seed 1 --loading 0 --k 10 &\
CUDA_VISIBLE_DEVICES=2 python offline_kd.py --env walker2d-random-v2  --bc_scale 0 --seed 1 --loading 0 --k 10 &\
CUDA_VISIBLE_DEVICES=3 python offline_kd.py --env walker2d-medium-v2  --bc_scale 0 --seed 1 --loading 0 --k 10 &\

# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



# CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --env hopper-random-v2

# pip3 install torch==1.10.0+cu110  -f https://download.pytorch.org/whl/cu113/torch_stable.html

# conda install pytorch  cudatoolkit=11.0 -c pytorch-lts -c nvidia

# conda install pytorch  cudatoolkit=11.3 -c pytorch

# conda install pytorch  cudatoolkit=11.3 -c pytorch

# pip3 install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

