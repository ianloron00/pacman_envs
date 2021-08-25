import numpy as np
from Environment import *

import gym, os
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
# import allStuff

name_model='discrete_pacman'
dir="tmp/"

env=PacmanEnv.make(zoom=2.0)

#### to test whether the standard functions are working ####
# episodes=10
# for eps in range(1, episodes+1):
#     # initializes each episode
    
#     state = env.reset() 
#     done = False
#     score = 0

#     while not done: # steps in episode
#         # env.render()
#         ####  env.action_space.sample()
#         action = env.getRandomAction()
#         new_state, reward, done, info = env.step(action)
#         score+=reward
#         # print("Reward: %s, Score: %s" %(reward, score))

#     print('Episode: {} Score: {}'.format(eps, score))
# env.close()


# applying stable_baselines models.
model=DQN('MlpPolicy', env, verbose=1)
if os.path.exists(dir + name_model+".zip"):
    model.load(name_model+".zip")


model.learn(total_timesteps=1e3) # 1e5
model.save(name_model)
evaluate_policy(model, env, n_eval_episodes=10, render=False, return_episode_rewards=False)
env.close()
