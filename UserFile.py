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

#### applying stable_baselines models.
"""
define replay-buffer:
buffer_size=1000000, learning_starts=50000

Change MlpPolicy from 64 x 64  to 128 x 64.
"""
model = DQN('MlpPolicy',env, buffer_size=20000, learning_starts=2000, exploration_fraction=0.3, exploration_initial_eps=1.0, exploration_final_eps=0.05, verbose=0)

if not os.path.exists(directory):
    os.mkdir(directory)

if os.path.exists(directory + name_model+".zip"):
    model.load(directory + name_model + ".zip")
    print("file loaded.")

if isTraining:
    timesteps = 3e4

    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=directory, name=name_model)
    model.learn(total_timesteps=timesteps, callback=callback)

    # plot rewards of training 
    plot_results([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Pacman")
    plt.savefig(directory + "DQN Pacman" + '.png')
    plt.show()

    # save model
    model.save(directory + name_model)

else:
    rewards = evaluate_policy(model, env, n_eval_episodes=10, render=False, return_episode_rewards=True)

    # plot results from policy evaluation
    scores=rewards[0]

    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(scores)) + 1, scores)
    plt.title("Pacman reward evaluation | DQN")
    plt.savefig(directory + "Pacman reward evaluation | DQN" + '.png')
    plt.show()
    plt.close()

env.close()

