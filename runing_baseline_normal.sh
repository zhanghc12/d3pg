CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy OurDDPG  --version $1  --tau 0.03 --env Walker2d-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=0 python main_dueling.py --policy OurDDPG  --version $1  --tau 0.03  --env Walker2d-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy OurDDPG  --version $1  --tau 0.03  --env Hopper-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy OurDDPG  --version $1  --tau 0.03  --env Hopper-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy OurDDPG  --version $1  --tau 0.03  --env HalfCheetah-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=2 python main_dueling.py --policy OurDDPG  --version $1  --tau 0.03  --env HalfCheetah-v2 --seed 1 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy OurDDPG  --version $1  --tau 0.03  --env Ant-v2 --seed 0 &\
CUDA_VISIBLE_DEVICES=3 python main_dueling.py --policy OurDDPG  --version $1  --tau 0.03  --env Ant-v2 --seed 1


# CUDA_VISIBLE_DEVICES=1 python main_dueling.py --policy DVPG  --version 2  --tau 0.1  --env Hopper-v2 --seed 1

157.7664096	0.344061005	0.035274994
365.5196848	0.685433451	0.052008806
241.4014084	0.591113425	0.042011506
123.434193	0.939428984	0.032509652
77.40956013	0.978945128	0.028802526

81.33240437	0.335550175	0.031219067
295.9112029	0.810819997	0.097150556
38.81752182	1.06689051	0.018155942
1388.738881	0.889782138	0.432932826
15.16047493	1.110705791	0.010887077


2.34520788	0.832421012	0.000156012
154.1487593	1.012656102	0.033223847
22.25848153	1.159475598	0.004493782
31.48586985	1.169670186	0.006503812