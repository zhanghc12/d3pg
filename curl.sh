CUDA_VISIBLE_DEVICES=0 python offline_curl.py --policy CURL  --version $1  --target_threshold $2 --env hopper-random-v2 --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=1 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env hopper-medium-v2  --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=2 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env hopper-medium-expert-v2  --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=3 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env hopper-medium-replay-v2 --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=4 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env hopper-expert-v2 --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=5 python offline_curl.py --policy CURL  --version $1  --target_threshold $2 --env halfcheetah-random-v2 --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=6 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env halfcheetah-medium-v2  --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=7 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env halfcheetah-medium-expert-v2  --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=0 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env halfcheetah-medium-replay-v2 --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=1 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env halfcheetah-expert-v2 --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=2 python offline_curl.py --policy CURL  --version $1  --target_threshold $2 --env walker2d-random-v2 --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=3 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env walker2d-medium-v2  --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=4 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env walker2d-medium-expert-v2  --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=5 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env walker2d-medium-replay-v2 --seed $3 --output_dim $4 --alpha $5 &\
CUDA_VISIBLE_DEVICES=6 python offline_curl.py --policy CURL  --version $1  --target_threshold $2  --env walker2d-expert-v2 --seed $3  --output_dim $4 --alpha $5



