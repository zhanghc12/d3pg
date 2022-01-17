import gym
import d4rl.gym_mujoco

env1 = gym.make('hopper-expert-v2')
env2 = gym.make('halfcheetah-expert-v2')
env3 = gym.make('walker2d-expert-v2')

for env in [env1, env2, env3]:
    print('env', env.observation_space.shape[0] + env.action_space.shape[0])


# hopper 0.07
# medium
# medium-replay be small
# medium-expert, expert must be smaller

# halfcheetah 0.005 * 23 = 0.115

# change another thing

# -> 0.1 not good -> 0.05, be 20
# [0.005, 0.07], must be 0.02, 0.05
# clip must be 50, 20 enough