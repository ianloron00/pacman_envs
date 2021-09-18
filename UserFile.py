"""
2- Treinar mais o pacman

--------------------------
- estado do pacman alterado. 
- Alterado gráfico de curva de aprendizado, para apresentar desvio padrao
"""

import numpy as np
from Environment import *

import gym, os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.results_plotter import plot_results

from Callback import *
from Graphs import *

# 'BoardState' | 'Selected'
extractor = 'BoardState'
name_model = 'discrete_pacman_' + extractor
directory = "tmp/"
isTraining = True

env = PacmanEnv.make(extractor=extractor, zoom=2.0)
env = Monitor(env, directory) 

"""
define replay-buffer:
buffer_size=1000000, learning_starts=50000
# UserWarning: This system does not have apparently enough memory to store the complete replay buffer 14.10GB > 12.06GB
Change MlpPolicy from 64 x 64  to 128 x 64.
"""
#### applying stable_baselines models.
model = DQN('MlpPolicy',env, tensorboard_log=directory, buffer_size=500000, 
            learning_starts=10000, exploration_fraction=0.5, exploration_initial_eps=1.0, 
            exploration_final_eps=0.05, verbose=0)

if not os.path.exists(directory):
    os.mkdir(directory)

if os.path.exists(directory + name_model+".zip"):
    model = model.load(directory + name_model + ".zip", env=env)
    
    print("file loaded.")

if isTraining:
    timesteps = 1e6

    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=directory, name=name_model)
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # plot rewards of training (self-made)
    plot_training([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Pacman")
    plt.savefig(directory + "iDQN " + name_model + '.png')
    plt.show()

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








""" (all)
Num timesteps: 10000
Best mean reward: -inf - Last mean reward per episode: -500.73
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 20000
Best mean reward: -500.73 - Last mean reward per episode: -593.43
Num timesteps: 30000
Best mean reward: -500.73 - Last mean reward per episode: -454.42
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 40000
Best mean reward: -454.42 - Last mean reward per episode: -485.41
Num timesteps: 50000
Best mean reward: -454.42 - Last mean reward per episode: -472.16
Num timesteps: 60000
Best mean reward: -454.42 - Last mean reward per episode: -428.34
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 70000
Best mean reward: -428.34 - Last mean reward per episode: -327.53
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 80000
Best mean reward: -327.53 - Last mean reward per episode: -364.71
Num timesteps: 90000
Best mean reward: -327.53 - Last mean reward per episode: -386.39
Num timesteps: 100000
Best mean reward: -327.53 - Last mean reward per episode: -289.93
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 110000
Best mean reward: -289.93 - Last mean reward per episode: -315.79
Num timesteps: 120000
Best mean reward: -289.93 - Last mean reward per episode: -310.68
Num timesteps: 130000
Best mean reward: -289.93 - Last mean reward per episode: -299.83
Num timesteps: 140000
Best mean reward: -289.93 - Last mean reward per episode: -305.35
Num timesteps: 150000
Best mean reward: -289.93 - Last mean reward per episode: -208.48
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 160000
Best mean reward: -208.48 - Last mean reward per episode: -164.11
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 170000
Best mean reward: -164.11 - Last mean reward per episode: -154.70
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 180000
Best mean reward: -154.70 - Last mean reward per episode: -157.64
Num timesteps: 190000
Best mean reward: -154.70 - Last mean reward per episode: -175.92
Num timesteps: 200000
Best mean reward: -154.70 - Last mean reward per episode: -107.38
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 210000
Best mean reward: -107.38 - Last mean reward per episode: -137.16
Num timesteps: 220000
Best mean reward: -107.38 - Last mean reward per episode: -61.70
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 230000
Best mean reward: -61.70 - Last mean reward per episode: -61.20
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 240000
Best mean reward: -61.20 - Last mean reward per episode: -66.15
Num timesteps: 250000
Best mean reward: -61.20 - Last mean reward per episode: -34.43
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 260000
Best mean reward: -34.43 - Last mean reward per episode: -104.79
Num timesteps: 270000
Best mean reward: -34.43 - Last mean reward per episode: -83.79
Num timesteps: 280000
Best mean reward: -34.43 - Last mean reward per episode: -194.51
Num timesteps: 290000
Best mean reward: -34.43 - Last mean reward per episode: -194.88
Num timesteps: 300000
Best mean reward: -34.43 - Last mean reward per episode: -0.96
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 310000
Best mean reward: -0.96 - Last mean reward per episode: 31.73
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 320000
Best mean reward: 31.73 - Last mean reward per episode: -261.98
Num timesteps: 330000
Best mean reward: 31.73 - Last mean reward per episode: -59.34
Num timesteps: 340000
Best mean reward: 31.73 - Last mean reward per episode: -186.81
Num timesteps: 350000
Best mean reward: 31.73 - Last mean reward per episode: -57.82
Num timesteps: 360000
Best mean reward: 31.73 - Last mean reward per episode: -5.39
Num timesteps: 370000
Best mean reward: 31.73 - Last mean reward per episode: 19.90
Num timesteps: 380000
Best mean reward: 31.73 - Last mean reward per episode: 43.35
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 390000
Best mean reward: 43.35 - Last mean reward per episode: 35.46
Num timesteps: 400000
Best mean reward: 43.35 - Last mean reward per episode: -44.52
Num timesteps: 410000
Best mean reward: 43.35 - Last mean reward per episode: 25.95
Num timesteps: 420000
Best mean reward: 43.35 - Last mean reward per episode: 6.99
Num timesteps: 430000
Best mean reward: 43.35 - Last mean reward per episode: 24.84
Num timesteps: 440000
Best mean reward: 43.35 - Last mean reward per episode: 20.57
Num timesteps: 450000
Best mean reward: 43.35 - Last mean reward per episode: 49.80
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 460000
Best mean reward: 49.80 - Last mean reward per episode: 49.01
Num timesteps: 470000
Best mean reward: 49.80 - Last mean reward per episode: 64.92
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 480000
Best mean reward: 64.92 - Last mean reward per episode: 27.57
Num timesteps: 490000
Best mean reward: 64.92 - Last mean reward per episode: 46.54
Num timesteps: 500000
Best mean reward: 64.92 - Last mean reward per episode: 35.34
Num timesteps: 510000
Best mean reward: 64.92 - Last mean reward per episode: 35.38
Num timesteps: 520000
Best mean reward: 64.92 - Last mean reward per episode: 24.95
Num timesteps: 530000
Best mean reward: 64.92 - Last mean reward per episode: 61.91
eps: 11000
(3, 'East', 0)
(3, 'East', 9)
(3, 'East', 9)
(3, 'East', 9)
(1, 'North', 9)
(1, 'North', 9)
(3, 'East', 9)
(3, 'East', 9)
(3, 'East', 9)
(1, 'North', 9)
(1, 'North', 9)
(4, 'West', 9)
(4, 'West', 9)
(0, 'Stop', 9)
(4, 'West', -10)
(1, 'North', 9)
(1, 'North', 9)
(4, 'West', 9)
(4, 'West', 9)
(4, 'West', 9)
(3, 'East', 9)
(4, 'West', -9)
(4, 'West', -9)
(3, 'East', 9)
(3, 'East', -9)
(3, 'East', 0)
(3, 'East', 0)
Num timesteps: 540000
Best mean reward: 64.92 - Last mean reward per episode: 58.76
Num timesteps: 550000
Best mean reward: 64.92 - Last mean reward per episode: 43.75
Num timesteps: 560000
Best mean reward: 64.92 - Last mean reward per episode: 56.08
Num timesteps: 570000
Best mean reward: 64.92 - Last mean reward per episode: 53.37
Num timesteps: 580000
Best mean reward: 64.92 - Last mean reward per episode: 10.60
Num timesteps: 590000
Best mean reward: 64.92 - Last mean reward per episode: -35.21
Num timesteps: 600000
Best mean reward: 64.92 - Last mean reward per episode: 80.43
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 610000
Best mean reward: 80.43 - Last mean reward per episode: 33.19
Num timesteps: 620000
Best mean reward: 80.43 - Last mean reward per episode: 32.87
Num timesteps: 630000
Best mean reward: 80.43 - Last mean reward per episode: 78.26
Num timesteps: 640000
Best mean reward: 80.43 - Last mean reward per episode: 3.25
Num timesteps: 650000
Best mean reward: 80.43 - Last mean reward per episode: 40.92
Num timesteps: 660000
Best mean reward: 80.43 - Last mean reward per episode: 77.58
Num timesteps: 670000
Best mean reward: 80.43 - Last mean reward per episode: 48.09
Num timesteps: 680000
Best mean reward: 80.43 - Last mean reward per episode: 86.27
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 690000
Best mean reward: 86.27 - Last mean reward per episode: 48.29
Num timesteps: 700000
Best mean reward: 86.27 - Last mean reward per episode: 47.32
Num timesteps: 710000
Best mean reward: 86.27 - Last mean reward per episode: 7.72
Num timesteps: 720000
Best mean reward: 86.27 - Last mean reward per episode: 7.30
Num timesteps: 730000
Best mean reward: 86.27 - Last mean reward per episode: 23.94
Num timesteps: 740000
Best mean reward: 86.27 - Last mean reward per episode: 6.32
Num timesteps: 750000
Best mean reward: 86.27 - Last mean reward per episode: -1.12
Num timesteps: 760000
Best mean reward: 86.27 - Last mean reward per episode: 25.59
Num timesteps: 770000
Best mean reward: 86.27 - Last mean reward per episode: 5.37
Num timesteps: 780000
Best mean reward: 86.27 - Last mean reward per episode: 35.36
Num timesteps: 790000
Best mean reward: 86.27 - Last mean reward per episode: 25.09
Num timesteps: 800000
Best mean reward: 86.27 - Last mean reward per episode: 36.44
Num timesteps: 810000
Best mean reward: 86.27 - Last mean reward per episode: 34.81
Num timesteps: 820000
Best mean reward: 86.27 - Last mean reward per episode: 18.54
Num timesteps: 830000
Best mean reward: 86.27 - Last mean reward per episode: 44.84
Num timesteps: 840000
Best mean reward: 86.27 - Last mean reward per episode: 45.67
Num timesteps: 850000
Best mean reward: 86.27 - Last mean reward per episode: 24.18
Num timesteps: 860000
Best mean reward: 86.27 - Last mean reward per episode: -20.79
Num timesteps: 870000
Best mean reward: 86.27 - Last mean reward per episode: -9.32
Num timesteps: 880000
Best mean reward: 86.27 - Last mean reward per episode: -31.81
Num timesteps: 890000
Best mean reward: 86.27 - Last mean reward per episode: -49.44
Num timesteps: 900000
Best mean reward: 86.27 - Last mean reward per episode: -56.30
Num timesteps: 910000
Best mean reward: 86.27 - Last mean reward per episode: -58.59
Num timesteps: 920000
Best mean reward: 86.27 - Last mean reward per episode: 8.59
Num timesteps: 930000
Best mean reward: 86.27 - Last mean reward per episode: 55.79
Num timesteps: 940000
Best mean reward: 86.27 - Last mean reward per episode: 59.73
eps: 22000
(3, 'East', 0)
(3, 'East', 9)
(3, 'East', 9)
(3, 'East', 9)
(1, 'North', 9)
(1, 'North', 9)
(4, 'West', 9)
(4, 'West', 9)
(4, 'West', 9)
(4, 'West', 9)
(4, 'West', 9)
(4, 'West', 9)
(4, 'West', 9)
(1, 'North', 9)
(1, 'North', 9)
(1, 'North', 9)
(1, 'North', 9)
(3, 'East', 9)
(3, 'East', 9)
(3, 'East', 9)
(3, 'East', 9)
(3, 'East', 9)
(3, 'East', 9)
(3, 'East', 9)
(3, 'East', 9)
(4, 'West', 9)
(4, 'West', -9)
(4, 'West', 0)
(4, 'West', 0)
Num timesteps: 950000
Best mean reward: 86.27 - Last mean reward per episode: 58.43
Num timesteps: 960000
Best mean reward: 86.27 - Last mean reward per episode: 40.21
Num timesteps: 970000
Best mean reward: 86.27 - Last mean reward per episode: 97.96
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 980000
Best mean reward: 97.96 - Last mean reward per episode: 82.26
Num timesteps: 990000
Best mean reward: 97.96 - Last mean reward per episode: 77.11
Num timesteps: 1000000
Best mean reward: 97.96 - Last mean reward per episode: 42.08
"""