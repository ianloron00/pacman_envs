import numpy as np
from Environment import *

import gym, os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt
# import allStuff

name_model='discrete_pacman'
directory="tmp/"
monitor = False

env=PacmanEnv.make(zoom=2.0)
if monitor: env = Monitor(env, directory)

#### to test whether the standard functions are working ####
# episodes=1
# for eps in range(1, episodes+1):
#     # initializes each episode
    
#     state = env.reset() 
#     done = False
#     score = 0

#     while not done: # steps in episode
#         env.render()
#         ####  env.action_space.sample()
#         action = env.getRandomAction()
#         new_state, reward, done, info = env.step(action)
#         score+=reward

#     print('Episode: {} Score: {}'.format(eps, score))
# env.close()


#### applying stable_baselines models.
model=DQN('MlpPolicy', env, verbose=1)

if not os.path.exists(directory):
    os.mkdir(directory)

if os.path.exists(directory + name_model+".zip"):
    model.load(directory + name_model+".zip")
else:
    timesteps = 1e5
    model.learn(total_timesteps=timesteps) 
    
    # plot learning rewarding results 
    plot_results([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Pacman")
    plt.show()

    # save model
    model.save(directory + name_model)

rewards = evaluate_policy(model, env, n_eval_episodes=10, render=(True and not monitor), return_episode_rewards=True)
print(rewards)

# plot results from policy evaluation
scores=rewards[0]
plt.plot(np.arange(len(scores)) + 1, scores)

plt.title("Pacman reward evaluation | DQN")
plt.show()

env.close()

