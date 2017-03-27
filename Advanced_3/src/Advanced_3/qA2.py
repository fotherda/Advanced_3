import gym
import numpy as np

env = gym.make('CartPole-v0')

# print(env.action_space)
#> Discrete(2)
# print(env.observation_space)
#> Box(4,)
# print(env.observation_space.high)
#> array([ 2.4       ,         inf,  0.20943951,         inf])
# print(env.observation_space.low)
#> array([-2.4       ,        -inf, -0.20943951,        -inf])

# from gym import spaces
# space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
# x = space.sample()
# assert space.contains(x)
# assert space.n == 8

MAX_EPISODES = 100
MAX_EPISODE_LENGTH = 300
GAMMA = 0.99


episode_length = np.zeros(MAX_EPISODES)
returns = np.zeros(MAX_EPISODES)

for i_episode in range(MAX_EPISODES):
    cum_reward = 0.0
    discount_factor = GAMMA
    observation = env.reset()
    for t in range(MAX_EPISODE_LENGTH):
        env.render()
#         print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            cum_reward -= 1.0 * discount_factor
            print("Episode {} finished after {} timesteps. Return {}".format(i_episode, t+1, cum_reward))
            break
        discount_factor *= GAMMA
        
    returns[i_episode] = cum_reward   
    episode_length[i_episode] = t+1   
    
print('mean episode length {} std {}'.format(np.mean(episode_length), np.std(episode_length)))
print('mean return {} std {}'.format(np.mean(returns), np.std(returns)))
        
        