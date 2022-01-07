CUDAVISIBLE_DEVICES=0 python main_nngp.py --num_points 100 &\
CUDAVISIBLE_DEVICES=1 python main_nngp.py --num_points 500 &\
CUDAVISIBLE_DEVICES=2 python main_nngp.py --num_points 1000 &\
CUDAVISIBLE_DEVICES=3 python main_nngp.py --num_points 2000