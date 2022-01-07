CUDA_VISIBLE_DEVICES=0 python main_nngp.py --num_points 100 &\
CUDA_VISIBLE_DEVICES=1 python main_nngp.py --num_points 500 &\
CUDA_VISIBLE_DEVICES=2 python main_nngp.py --num_points 1000 &\
CUDA_VISIBLE_DEVICES=3 python main_nngp.py --num_points 2000