"""
1- Alterar gráfico de curva de aprendizado, para apresentar desvio padrao

2- Treinar mais o pacman

--------------------------
- estado do pacman alterado. 
"""


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

# 'BoardState' | 'Selected'
extractor = 'Selected'
name_model = 'discrete_pacman_' + extractor
directory = "tmp/"
isTraining = True

env = PacmanEnv.make(extractor=extractor, zoom=2.0)
env = Monitor(env, directory) 

"""
define replay-buffer:
buffer_size=1000000, learning_starts=50000

Change MlpPolicy from 64 x 64  to 128 x 64.
"""
#### applying stable_baselines models.
model = DQN('MlpPolicy',env, tensorboard_log=directory, buffer_size=1000000, 
            learning_starts=10000, exploration_fraction=0.5, exploration_initial_eps=1.0, 
            exploration_final_eps=0.05, verbose=0)

if not os.path.exists(directory):
    os.mkdir(directory)

if os.path.exists(directory + name_model+".zip"):
    model = model.load(directory + name_model + ".zip", env=env)
    
    print("file loaded.")

if isTraining:
    timesteps = 6e2

    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=directory, name=name_model)
    model.learn(total_timesteps=timesteps, callback=callback)

    # plot rewards of training 
    plot_results([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Pacman")
    plt.savefig(directory + "DQN " + name_model + '.png')
    plt.show()

    ## save model, only if there is no callback.
    # model.save(directory + name_model)

else:
    rewards = evaluate_policy(model, env, n_eval_episodes=100, render=True, return_episode_rewards=True)

    # plot results from policy evaluation
    scores=rewards[0]

    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(scores)) + 1, scores)
    plt.title("Pacman reward evaluation | DQN")
    plt.savefig(directory + "Pacman R eval. | extractor " + extractor + " | DQN" + '.png')
    plt.show()
    plt.close()

env.close()

"""
tensorboard --logdir tmp/figure

https://www.tensorflow.org/tensorboard/get_started
"""