CUDAVISIBLE_DEVICE=1 python main_nngp.py --num_points 100 &\
CUDAVISIBLE_DEVICE=2 python main_nngp.py --num_points 1000 &\
CUDAVISIBLE_DEVICE=3 python main_nngp.py --num_points 2000