import gym

env = gym.make('HalfCheetah-v2')
env.reset()

for i in range(10000):
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    if done:
        print(i)