CUDA_VISIBLE_DEVICES=0 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.0001  --env  Walker2d-v2  &\
CUDA_VISIBLE_DEVICES=1 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.001   --env  Walker2d-v2  &\
CUDA_VISIBLE_DEVICES=2 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.01   --env  Walker2d-v2  &\
CUDA_VISIBLE_DEVICES=3 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.1   --env  Walker2d-v2  &\
CUDA_VISIBLE_DEVICES=4 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.0001  --env  Hopper-v2  &\
CUDA_VISIBLE_DEVICES=5 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.001   --env  Hopper-v2  &\
CUDA_VISIBLE_DEVICES=6 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.01   --env  Hopper-v2  &\
CUDA_VISIBLE_DEVICES=7 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.1   --env  Hopper-v2  &\
CUDA_VISIBLE_DEVICES=0 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.0001  --env  HalfCheetah-v2  &\
CUDA_VISIBLE_DEVICES=1 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.001   --env  HalfCheetah-v2  &\
CUDA_VISIBLE_DEVICES=2 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.01   --env  HalfCheetah-v2  &\
CUDA_VISIBLE_DEVICES=3 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.1   --env  HalfCheetah-v2  &\
CUDA_VISIBLE_DEVICES=4 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.0001  --env  Ant-v2  &\
CUDA_VISIBLE_DEVICES=5 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.001   --env  Ant-v2  &\
CUDA_VISIBLE_DEVICES=6 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.01   --env  Ant-v2  &\
CUDA_VISIBLE_DEVICES=7 python main_dnpg_v3.py --policy D4PG  --version $1  --target_threshold 0.1   --env  Ant-v2


# sh eval_ac_dnpg_8gpu.sh






