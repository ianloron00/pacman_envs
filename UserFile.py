import numpy as np
from Environment import *

import gym, os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
import matplotlib.pyplot as plt
from Callback import *


name_model='discrete_pacman'
directory="tmp/"
isTraining = False

env=PacmanEnv.make(zoom=2.0)
env=Monitor(env, directory) 

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
model=DQN('MlpPolicy', env, exploration_fraction=0.2, exploration_initial_eps=1.0, exploration_final_eps=0.03, verbose=0)

if not os.path.exists(directory):
    os.mkdir(directory)

if os.path.exists(directory + name_model+".zip"):
    print("file loaded.")
    model.load(directory + name_model + ".zip")

if isTraining:
    timesteps = 1e5

    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=directory, name=name_model)
    model.learn(total_timesteps=timesteps, callback=callback)

    # plot rewards of training 
    plot_results([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Pacman")
    plt.savefig(directory + "DQN Pacman" + '.png')
    plt.show()

    # save model
    model.save(directory + name_model)

else:
    rewards = evaluate_policy(model, env, n_eval_episodes=10, render=True, return_episode_rewards=True)

    # plot results from policy evaluation
    scores=rewards[0]

    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(scores)) + 1, scores)
    plt.title("Pacman reward evaluation | DQN")
    plt.savefig(directory + "Pacman reward evaluation | DQN" + '.png')
    plt.show()
    plt.close()

env.close()

