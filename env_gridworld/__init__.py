from gym.envs.registration import registry, register, make, spec
from env_gridworld.gridworld import GridWorld
register(
    id='GridWorld-v0',
    entry_point='env_gridworld:GridWorld',
    max_episode_steps=100,
    # reward_threshold=90.0,
)