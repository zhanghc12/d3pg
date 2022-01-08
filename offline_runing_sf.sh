CUDA_VISIBLE_DEVICES=0 python offline_sf.py --env hopper-expert-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python offline_sf.py --env hopper-medium-v2  --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python offline_sf.py --env walker2d-expert-v2  --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python offline_sf.py --env hopper-medium-expert-v2 --seed 1

# sh offline_rl_large_scale_single_seed_new.sh 31  1.0 -1 0.2  0.05



# CUDA_VISIBLE_DEVICES=0 python offline_tqc.py --env hopper-random-v2

# pip3 install torch==1.10.0+cu110  -f https://download.pytorch.org/whl/cu113/torch_stable.html

# conda install pytorch  cudatoolkit=11.0 -c pytorch-lts -c nvidia

# conda install pytorch  cudatoolkit=11.3 -c pytorch

# conda install pytorch  cudatoolkit=11.3 -c pytorch

# pip3 install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

