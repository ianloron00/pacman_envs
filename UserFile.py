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
isTraining = False

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
            learning_starts=10000, exploration_fraction=1.0, exploration_initial_eps=0.326, 
            exploration_final_eps=0.05, verbose=0)

if not os.path.exists(directory):
    os.mkdir(directory)

if os.path.exists(directory + name_model+".zip"):
    model = model.load(directory + name_model + ".zip", env=env)
    print("model loaded.")

elif os.path.exists(directory + name_model + "/" + name_model+".zip"):
    model = model.load(directory + name_model + "/" + name_model+".zip", env=env)
    print("callback loaded.")

if isTraining:
    timesteps = 1.63e6

    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=directory, name=name_model)
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # plot rewards of training (self-made)
    plot_training([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Pacman")
    plt.savefig(directory + "iDQN_halh " + name_model + '.png')
    plt.show()

    # plot rewards of training 
    plot_results([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Pacman")
    plt.savefig(directory + "DQN_half " + name_model + '.png')
    plt.show()

    ## save model, only if there is no callback.
    # model.save(directory + name_model)

else:
    rewards = evaluate_policy(model, env, n_eval_episodes=100, render=False, return_episode_rewards=True)

    # plot results from policy evaluation
    scores=rewards[0]

    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(scores)) + 1, scores)
    plt.title("Pacman reward evaluation | DQN")
    plt.savefig(directory + "Pacman R eval. | extractor<test R> " + extractor + " | DQN" + '.png')
    plt.show()
    plt.close()

env.close()



"""
BoardState
Num timesteps: 10000
Best mean reward: -inf - Last mean reward per episode: -307.88
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 20000
Best mean reward: -307.88 - Last mean reward per episode: -333.93
Num timesteps: 30000
Best mean reward: -307.88 - Last mean reward per episode: -363.50
Num timesteps: 40000
Best mean reward: -307.88 - Last mean reward per episode: -322.04
Num timesteps: 50000
Best mean reward: -307.88 - Last mean reward per episode: -354.17
Num timesteps: 60000
Best mean reward: -307.88 - Last mean reward per episode: -327.14
Num timesteps: 70000
Best mean reward: -307.88 - Last mean reward per episode: -404.14
Num timesteps: 80000
Best mean reward: -307.88 - Last mean reward per episode: -356.71
Num timesteps: 90000
Best mean reward: -307.88 - Last mean reward per episode: -347.26
Num timesteps: 100000
Best mean reward: -307.88 - Last mean reward per episode: -354.91
Num timesteps: 110000
Best mean reward: -307.88 - Last mean reward per episode: -277.34
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 120000
Best mean reward: -277.34 - Last mean reward per episode: -325.15
Num timesteps: 130000
Best mean reward: -277.34 - Last mean reward per episode: -276.48
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 140000
Best mean reward: -276.48 - Last mean reward per episode: -303.19
Num timesteps: 150000
Best mean reward: -276.48 - Last mean reward per episode: -300.16
Num timesteps: 160000
Best mean reward: -276.48 - Last mean reward per episode: -288.49
Num timesteps: 170000
Best mean reward: -276.48 - Last mean reward per episode: -311.57
Num timesteps: 180000
Best mean reward: -276.48 - Last mean reward per episode: -320.81
Num timesteps: 190000
Best mean reward: -276.48 - Last mean reward per episode: -337.40
Num timesteps: 200000
Best mean reward: -276.48 - Last mean reward per episode: -359.07
Num timesteps: 210000
Best mean reward: -276.48 - Last mean reward per episode: -277.81
Num timesteps: 220000
Best mean reward: -276.48 - Last mean reward per episode: -498.25
Num timesteps: 230000
Best mean reward: -276.48 - Last mean reward per episode: -258.18
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 240000
Best mean reward: -258.18 - Last mean reward per episode: -303.07
Num timesteps: 250000
Best mean reward: -258.18 - Last mean reward per episode: -367.03
Num timesteps: 260000
Best mean reward: -258.18 - Last mean reward per episode: -424.98
Num timesteps: 270000
Best mean reward: -258.18 - Last mean reward per episode: -310.32
Num timesteps: 280000
Best mean reward: -258.18 - Last mean reward per episode: -341.65
Num timesteps: 290000
Best mean reward: -258.18 - Last mean reward per episode: -339.45
Num timesteps: 300000
Best mean reward: -258.18 - Last mean reward per episode: -347.50
Num timesteps: 310000
Best mean reward: -258.18 - Last mean reward per episode: -411.76
Num timesteps: 320000
Best mean reward: -258.18 - Last mean reward per episode: -310.25
Num timesteps: 330000
Best mean reward: -258.18 - Last mean reward per episode: -309.58
Num timesteps: 340000
Best mean reward: -258.18 - Last mean reward per episode: -287.15
Num timesteps: 350000
Best mean reward: -258.18 - Last mean reward per episode: -309.19
Num timesteps: 360000
Best mean reward: -258.18 - Last mean reward per episode: -317.42
Num timesteps: 370000
Best mean reward: -258.18 - Last mean reward per episode: -351.38
Num timesteps: 380000
Best mean reward: -258.18 - Last mean reward per episode: -444.62
Num timesteps: 390000
Best mean reward: -258.18 - Last mean reward per episode: -304.06
Num timesteps: 400000
Best mean reward: -258.18 - Last mean reward per episode: -272.52
Num timesteps: 410000
Best mean reward: -258.18 - Last mean reward per episode: -298.37
Num timesteps: 420000
Best mean reward: -258.18 - Last mean reward per episode: -301.70
Num timesteps: 430000
Best mean reward: -258.18 - Last mean reward per episode: -278.98
Num timesteps: 440000
Best mean reward: -258.18 - Last mean reward per episode: -324.56
Num timesteps: 450000
Best mean reward: -258.18 - Last mean reward per episode: -335.81
Num timesteps: 460000
Best mean reward: -258.18 - Last mean reward per episode: -296.20
Num timesteps: 470000
Best mean reward: -258.18 - Last mean reward per episode: -332.67
Num timesteps: 480000
Best mean reward: -258.18 - Last mean reward per episode: -289.90
Num timesteps: 490000
Best mean reward: -258.18 - Last mean reward per episode: -321.72
Num timesteps: 500000
Best mean reward: -258.18 - Last mean reward per episode: -315.83
Num timesteps: 510000
Best mean reward: -258.18 - Last mean reward per episode: -308.25
Num timesteps: 520000
Best mean reward: -258.18 - Last mean reward per episode: -454.24
Num timesteps: 530000
Best mean reward: -258.18 - Last mean reward per episode: -410.52
Num timesteps: 540000
Best mean reward: -258.18 - Last mean reward per episode: -326.46
Num timesteps: 550000
Best mean reward: -258.18 - Last mean reward per episode: -296.95
Num timesteps: 560000
Best mean reward: -258.18 - Last mean reward per episode: -309.50
Num timesteps: 570000
Best mean reward: -258.18 - Last mean reward per episode: -278.81
Num timesteps: 580000
Best mean reward: -258.18 - Last mean reward per episode: -253.07
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 590000
Best mean reward: -253.07 - Last mean reward per episode: -270.81
Num timesteps: 600000
Best mean reward: -253.07 - Last mean reward per episode: -288.58
Num timesteps: 610000
Best mean reward: -253.07 - Last mean reward per episode: -279.46
Num timesteps: 620000
Best mean reward: -253.07 - Last mean reward per episode: -344.89
Num timesteps: 630000
Best mean reward: -253.07 - Last mean reward per episode: -301.35
Num timesteps: 640000
Best mean reward: -253.07 - Last mean reward per episode: -322.06
Num timesteps: 650000
Best mean reward: -253.07 - Last mean reward per episode: -278.36
Num timesteps: 660000
Best mean reward: -253.07 - Last mean reward per episode: -298.49
Num timesteps: 670000
Best mean reward: -253.07 - Last mean reward per episode: -268.45
Num timesteps: 680000
Best mean reward: -253.07 - Last mean reward per episode: -256.26
Num timesteps: 690000
Best mean reward: -253.07 - Last mean reward per episode: -218.43
Saving new best model to tmp/discrete_pacman_BoardState
eps: 11000
(3, 'East', 0)
(2, 'Stop', 6)
(2, 'Stop', -1)
(0, 'Stop', -1.5)
(4, 'West', -1.5)
(2, 'Stop', 0)
(3, 'East', -1)
(3, 'East', 0)
(2, 'Stop', 7)
(4, 'West', -2)
(1, 'Stop', 0)
(2, 'Stop', -2)
(0, 'Stop', -3.0)
(1, 'Stop', -3.0)
(3, 'East', -3.0)
(0, 'Stop', 0)
(2, 'Stop', -2)
(0, 'Stop', -3.0)
(3, 'East', -3.0)
(1, 'Stop', 8)
(1, 'Stop', -3)
(3, 'East', -4.5)
(0, 'Stop', 9)
(2, 'Stop', -4)
(2, 'Stop', -6.0)
(1, 'North', -6.0)
(3, 'Stop', 10)
(3, 'Stop', -5)
(0, 'Stop', -7.5)
(3, 'Stop', -7.5)
(1, 'North', -7.5)
(0, 'Stop', 11)
(2, 'South', -6)
(3, 'Stop', 0)
(4, 'Stop', -6)
(3, 'Stop', -9.0)
(1, 'North', -9.0)
(1, 'North', 0)
(1, 'North', 12)
(2, 'South', 13)
(0, 'Stop', -4.0)
(0, 'Stop', -8)
(3, 'Stop', -12.0)
(2, 'South', -12.0)
(1, 'North', 0)
(4, 'Stop', -4.0)
(1, 'North', -8)
(4, 'Stop', 0)
(1, 'North', -8)
(4, 'Stop', 14)
(1, 'North', -9)
(1, 'North', 15)
(4, 'Stop', 16)
(2, 'South', -11)
(3, 'East', 0)
(0, 'Stop', 17)
(3, 'East', -12)
(0, 'Stop', 18)
(0, 'Stop', -13)
(3, 'East', -19.5)
(3, 'Stop', 19)
(0, 'Stop', -14)
(4, 'West', -21.0)
(3, 'East', 0)
(3, 'Stop', -7.0)
(0, 'Stop', -14)
(2, 'South', -21.0)
(1, 'North', 20)
(2, 'South', -7.5)
Num timesteps: 700000
Best mean reward: -218.43 - Last mean reward per episode: -261.52
Num timesteps: 710000
Best mean reward: -218.43 - Last mean reward per episode: -410.74
Num timesteps: 720000
Best mean reward: -218.43 - Last mean reward per episode: -271.38
Num timesteps: 730000
Best mean reward: -218.43 - Last mean reward per episode: -245.96
Num timesteps: 740000
Best mean reward: -218.43 - Last mean reward per episode: -279.23
Num timesteps: 750000
Best mean reward: -218.43 - Last mean reward per episode: -223.19
Num timesteps: 760000
Best mean reward: -218.43 - Last mean reward per episode: -438.11
Num timesteps: 770000
Best mean reward: -218.43 - Last mean reward per episode: -331.17
Num timesteps: 780000
Best mean reward: -218.43 - Last mean reward per episode: -335.94
Num timesteps: 790000
Best mean reward: -218.43 - Last mean reward per episode: -244.91
Num timesteps: 800000
Best mean reward: -218.43 - Last mean reward per episode: -373.18
Num timesteps: 810000
Best mean reward: -218.43 - Last mean reward per episode: -358.51
Num timesteps: 820000
Best mean reward: -218.43 - Last mean reward per episode: -388.19
Num timesteps: 830000
Best mean reward: -218.43 - Last mean reward per episode: -308.21
Num timesteps: 840000
Best mean reward: -218.43 - Last mean reward per episode: -284.23
Num timesteps: 850000
Best mean reward: -218.43 - Last mean reward per episode: -280.42
Num timesteps: 860000
Best mean reward: -218.43 - Last mean reward per episode: -261.38
Num timesteps: 870000
Best mean reward: -218.43 - Last mean reward per episode: -283.77
Num timesteps: 880000
Best mean reward: -218.43 - Last mean reward per episode: -305.04
Num timesteps: 890000
Best mean reward: -218.43 - Last mean reward per episode: -291.79
Num timesteps: 900000
Best mean reward: -218.43 - Last mean reward per episode: -274.68
Num timesteps: 910000
Best mean reward: -218.43 - Last mean reward per episode: -317.10
Num timesteps: 920000
Best mean reward: -218.43 - Last mean reward per episode: -336.03
Num timesteps: 930000
Best mean reward: -218.43 - Last mean reward per episode: -248.38
Num timesteps: 940000
Best mean reward: -218.43 - Last mean reward per episode: -197.85
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 950000
Best mean reward: -197.85 - Last mean reward per episode: -289.34
Num timesteps: 960000
Best mean reward: -197.85 - Last mean reward per episode: -304.62
Num timesteps: 970000
Best mean reward: -197.85 - Last mean reward per episode: -290.80
Num timesteps: 980000
Best mean reward: -197.85 - Last mean reward per episode: -283.86
Num timesteps: 990000
Best mean reward: -197.85 - Last mean reward per episode: -252.04
Num timesteps: 1000000
Best mean reward: -197.85 - Last mean reward per episode: -256.96
Num timesteps: 1010000
Best mean reward: -197.85 - Last mean reward per episode: -271.33
Num timesteps: 1020000
Best mean reward: -197.85 - Last mean reward per episode: -263.44
Num timesteps: 1030000
Best mean reward: -197.85 - Last mean reward per episode: -198.56
Num timesteps: 1040000
Best mean reward: -197.85 - Last mean reward per episode: -206.20
Num timesteps: 1050000
Best mean reward: -197.85 - Last mean reward per episode: -261.00
Num timesteps: 1060000
Best mean reward: -197.85 - Last mean reward per episode: -351.81
Num timesteps: 1070000
Best mean reward: -197.85 - Last mean reward per episode: -239.97
Num timesteps: 1080000
Best mean reward: -197.85 - Last mean reward per episode: -333.44
Num timesteps: 1090000
Best mean reward: -197.85 - Last mean reward per episode: -216.78
Num timesteps: 1100000
Best mean reward: -197.85 - Last mean reward per episode: -330.98
Num timesteps: 1110000
Best mean reward: -197.85 - Last mean reward per episode: -310.54
Num timesteps: 1120000
Best mean reward: -197.85 - Last mean reward per episode: -264.56
Num timesteps: 1130000
Best mean reward: -197.85 - Last mean reward per episode: -265.83
Num timesteps: 1140000
Best mean reward: -197.85 - Last mean reward per episode: -201.34
Num timesteps: 1150000
Best mean reward: -197.85 - Last mean reward per episode: -278.72
Num timesteps: 1160000
Best mean reward: -197.85 - Last mean reward per episode: -224.10
Num timesteps: 1170000
Best mean reward: -197.85 - Last mean reward per episode: -230.62
Num timesteps: 1180000
Best mean reward: -197.85 - Last mean reward per episode: -256.12
Num timesteps: 1190000
Best mean reward: -197.85 - Last mean reward per episode: -219.05
Num timesteps: 1200000
Best mean reward: -197.85 - Last mean reward per episode: -324.06
Num timesteps: 1210000
Best mean reward: -197.85 - Last mean reward per episode: -224.28
Num timesteps: 1220000
Best mean reward: -197.85 - Last mean reward per episode: -226.78
Num timesteps: 1230000
Best mean reward: -197.85 - Last mean reward per episode: -290.51
Num timesteps: 1240000
Best mean reward: -197.85 - Last mean reward per episode: -241.16
Num timesteps: 1250000
Best mean reward: -197.85 - Last mean reward per episode: -189.11
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1260000
Best mean reward: -189.11 - Last mean reward per episode: -209.19
Num timesteps: 1270000
Best mean reward: -189.11 - Last mean reward per episode: -220.19
Num timesteps: 1280000
Best mean reward: -189.11 - Last mean reward per episode: -189.57
Num timesteps: 1290000
Best mean reward: -189.11 - Last mean reward per episode: -205.31
Num timesteps: 1300000
Best mean reward: -189.11 - Last mean reward per episode: -336.80
Num timesteps: 1310000
Best mean reward: -189.11 - Last mean reward per episode: -217.78
Num timesteps: 1320000
Best mean reward: -189.11 - Last mean reward per episode: -217.25
Num timesteps: 1330000
Best mean reward: -189.11 - Last mean reward per episode: -280.23
Num timesteps: 1340000
Best mean reward: -189.11 - Last mean reward per episode: -235.12
Num timesteps: 1350000
Best mean reward: -189.11 - Last mean reward per episode: -390.69
eps: 22000
(2, 'Stop', 0)
(4, 'West', 0)
(4, 'West', 6)
(2, 'Stop', 7)
(0, 'Stop', -2)
(4, 'West', -3.0)
(4, 'Stop', 8)
(1, 'North', -3)
(1, 'North', 9)
(1, 'North', 10)
(1, 'North', 11)
(3, 'Stop', 12)
(1, 'North', -7)
(1, 'North', 13)
(1, 'North', 14)
(0, 'Stop', 15)
(1, 'North', -10)
(1, 'Stop', 16)
(3, 'East', -11)
(2, 'Stop', 17)
(1, 'Stop', -12)
(4, 'West', -18.0)
(3, 'East', 0)
(2, 'Stop', -6.0)
(3, 'East', -12)
(0, 'Stop', 18)
(1, 'Stop', -13)
(3, 'East', -19.5)
(3, 'East', 19)
(2, 'Stop', 20)
(3, 'East', -15)
(3, 'East', 21)
(4, 'West', 22)
(4, 'West', -8.5)
(1, 'Stop', 0)
(2, 'Stop', -17)
(0, 'Stop', -25.5)
(4, 'West', -25.5)
(4, 'West', 0)
(2, 'Stop', 0)
Num timesteps: 1360000
Best mean reward: -189.11 - Last mean reward per episode: -377.93
Num timesteps: 1370000
Best mean reward: -189.11 - Last mean reward per episode: -254.04
Num timesteps: 1380000
Best mean reward: -189.11 - Last mean reward per episode: -260.22
Num timesteps: 1390000
Best mean reward: -189.11 - Last mean reward per episode: -351.88
Num timesteps: 1400000
Best mean reward: -189.11 - Last mean reward per episode: -359.57
Num timesteps: 1410000
Best mean reward: -189.11 - Last mean reward per episode: -188.67
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1420000
Best mean reward: -188.67 - Last mean reward per episode: -251.85
Num timesteps: 1430000
Best mean reward: -188.67 - Last mean reward per episode: -244.81
Num timesteps: 1440000
Best mean reward: -188.67 - Last mean reward per episode: -248.71
Num timesteps: 1450000
Best mean reward: -188.67 - Last mean reward per episode: -261.33
Num timesteps: 1460000
Best mean reward: -188.67 - Last mean reward per episode: -273.27
Num timesteps: 1470000
Best mean reward: -188.67 - Last mean reward per episode: -188.83
Num timesteps: 1480000
Best mean reward: -188.67 - Last mean reward per episode: -248.46
Num timesteps: 1490000
Best mean reward: -188.67 - Last mean reward per episode: -251.93
Num timesteps: 1500000
Best mean reward: -188.67 - Last mean reward per episode: -215.03
Num timesteps: 1510000
Best mean reward: -188.67 - Last mean reward per episode: -248.19
Num timesteps: 1520000
Best mean reward: -188.67 - Last mean reward per episode: -188.59
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1530000
Best mean reward: -188.59 - Last mean reward per episode: -247.45
Num timesteps: 1540000
Best mean reward: -188.59 - Last mean reward per episode: -191.06
Num timesteps: 1550000
Best mean reward: -188.59 - Last mean reward per episode: -291.89
Num timesteps: 1560000
Best mean reward: -188.59 - Last mean reward per episode: -240.60
Num timesteps: 1570000
Best mean reward: -188.59 - Last mean reward per episode: -243.38
Num timesteps: 1580000
Best mean reward: -188.59 - Last mean reward per episode: -314.98
Num timesteps: 1590000
Best mean reward: -188.59 - Last mean reward per episode: -217.47
Num timesteps: 1600000
Best mean reward: -188.59 - Last mean reward per episode: -232.34
Num timesteps: 1610000
Best mean reward: -188.59 - Last mean reward per episode: -248.78
Num timesteps: 1620000
Best mean reward: -188.59 - Last mean reward per episode: -205.75
Num timesteps: 1630000
Best mean reward: -188.59 - Last mean reward per episode: -236.15
Num timesteps: 1640000
Best mean reward: -188.59 - Last mean reward per episode: -155.86
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1650000
Best mean reward: -155.86 - Last mean reward per episode: -148.71
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1660000
Best mean reward: -148.71 - Last mean reward per episode: -241.01
Num timesteps: 1670000
Best mean reward: -148.71 - Last mean reward per episode: -194.62
Num timesteps: 1680000
Best mean reward: -148.71 - Last mean reward per episode: -259.25
Num timesteps: 1690000
Best mean reward: -148.71 - Last mean reward per episode: -171.97
Num timesteps: 1700000
Best mean reward: -148.71 - Last mean reward per episode: -186.59
Num timesteps: 1710000
Best mean reward: -148.71 - Last mean reward per episode: -154.93
Num timesteps: 1720000
Best mean reward: -148.71 - Last mean reward per episode: -144.12
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1730000
Best mean reward: -144.12 - Last mean reward per episode: -155.50
Num timesteps: 1740000
Best mean reward: -144.12 - Last mean reward per episode: -153.28
Num timesteps: 1750000
Best mean reward: -144.12 - Last mean reward per episode: -217.69
Num timesteps: 1760000
Best mean reward: -144.12 - Last mean reward per episode: -222.69
Killed
"""
"""
BoardState
file loaded.
Num timesteps: 10000
Best mean reward: -inf - Last mean reward per episode: -397.16
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 20000
Best mean reward: -397.16 - Last mean reward per episode: -284.85
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 30000
Best mean reward: -284.85 - Last mean reward per episode: -333.06
Num timesteps: 40000
Best mean reward: -284.85 - Last mean reward per episode: -353.43
Num timesteps: 50000
Best mean reward: -284.85 - Last mean reward per episode: -301.88
Num timesteps: 60000
Best mean reward: -284.85 - Last mean reward per episode: -387.21
Num timesteps: 70000
Best mean reward: -284.85 - Last mean reward per episode: -322.04
Num timesteps: 80000
Best mean reward: -284.85 - Last mean reward per episode: -349.39
Num timesteps: 90000
Best mean reward: -284.85 - Last mean reward per episode: -345.01
Num timesteps: 100000
Best mean reward: -284.85 - Last mean reward per episode: -326.10
Num timesteps: 110000
Best mean reward: -284.85 - Last mean reward per episode: -339.63
Num timesteps: 120000
Best mean reward: -284.85 - Last mean reward per episode: -268.96
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 130000
Best mean reward: -268.96 - Last mean reward per episode: -315.26
Num timesteps: 140000
Best mean reward: -268.96 - Last mean reward per episode: -289.90
Num timesteps: 150000
Best mean reward: -268.96 - Last mean reward per episode: -358.57
Num timesteps: 160000
Best mean reward: -268.96 - Last mean reward per episode: -365.33
Num timesteps: 170000
Best mean reward: -268.96 - Last mean reward per episode: -333.74
Num timesteps: 180000
Best mean reward: -268.96 - Last mean reward per episode: -331.31
Num timesteps: 190000
Best mean reward: -268.96 - Last mean reward per episode: -296.08
Num timesteps: 200000
Best mean reward: -268.96 - Last mean reward per episode: -393.65
Num timesteps: 210000
Best mean reward: -268.96 - Last mean reward per episode: -257.71
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 220000
Best mean reward: -257.71 - Last mean reward per episode: -344.85
Num timesteps: 230000
Best mean reward: -257.71 - Last mean reward per episode: -300.67
Num timesteps: 240000
Best mean reward: -257.71 - Last mean reward per episode: -324.01
Num timesteps: 250000
Best mean reward: -257.71 - Last mean reward per episode: -278.88
Num timesteps: 260000
Best mean reward: -257.71 - Last mean reward per episode: -273.93
Num timesteps: 270000
Best mean reward: -257.71 - Last mean reward per episode: -239.19
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 280000
Best mean reward: -239.19 - Last mean reward per episode: -241.47
Num timesteps: 290000
Best mean reward: -239.19 - Last mean reward per episode: -330.04
Num timesteps: 300000
Best mean reward: -239.19 - Last mean reward per episode: -285.46
Num timesteps: 310000
Best mean reward: -239.19 - Last mean reward per episode: -316.06
Num timesteps: 320000
Best mean reward: -239.19 - Last mean reward per episode: -262.70
Num timesteps: 330000
Best mean reward: -239.19 - Last mean reward per episode: -236.28
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 340000
Best mean reward: -236.28 - Last mean reward per episode: -219.22
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 350000
Best mean reward: -219.22 - Last mean reward per episode: -298.33
Num timesteps: 360000
Best mean reward: -219.22 - Last mean reward per episode: -284.05
Num timesteps: 370000
Best mean reward: -219.22 - Last mean reward per episode: -215.12
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 380000
Best mean reward: -215.12 - Last mean reward per episode: -310.83
Num timesteps: 390000
Best mean reward: -215.12 - Last mean reward per episode: -196.86
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 400000
Best mean reward: -196.86 - Last mean reward per episode: -265.15
Num timesteps: 410000
Best mean reward: -196.86 - Last mean reward per episode: -174.49
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 420000
Best mean reward: -174.49 - Last mean reward per episode: -232.17
Num timesteps: 430000
Best mean reward: -174.49 - Last mean reward per episode: -196.16
Num timesteps: 440000
Best mean reward: -174.49 - Last mean reward per episode: -225.76
Num timesteps: 450000
Best mean reward: -174.49 - Last mean reward per episode: -312.93
Num timesteps: 460000
Best mean reward: -174.49 - Last mean reward per episode: -338.38
Num timesteps: 470000
Best mean reward: -174.49 - Last mean reward per episode: -209.53
Num timesteps: 480000
Best mean reward: -174.49 - Last mean reward per episode: -212.53
Num timesteps: 490000
Best mean reward: -174.49 - Last mean reward per episode: -248.40
Num timesteps: 500000
Best mean reward: -174.49 - Last mean reward per episode: -181.88
Num timesteps: 510000
Best mean reward: -174.49 - Last mean reward per episode: -247.69
Num timesteps: 520000
Best mean reward: -174.49 - Last mean reward per episode: -192.09
Num timesteps: 530000
Best mean reward: -174.49 - Last mean reward per episode: -269.50
Num timesteps: 540000
Best mean reward: -174.49 - Last mean reward per episode: -198.80
Num timesteps: 550000
Best mean reward: -174.49 - Last mean reward per episode: -216.79
Num timesteps: 560000
Best mean reward: -174.49 - Last mean reward per episode: -255.21
Num timesteps: 570000
Best mean reward: -174.49 - Last mean reward per episode: -195.47
Num timesteps: 580000
Best mean reward: -174.49 - Last mean reward per episode: -154.72
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 590000
Best mean reward: -154.72 - Last mean reward per episode: -211.68
Num timesteps: 600000
Best mean reward: -154.72 - Last mean reward per episode: -306.62
Num timesteps: 610000
Best mean reward: -154.72 - Last mean reward per episode: -227.84
Num timesteps: 620000
Best mean reward: -154.72 - Last mean reward per episode: -123.16
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 630000
Best mean reward: -123.16 - Last mean reward per episode: -190.49
Num timesteps: 640000
Best mean reward: -123.16 - Last mean reward per episode: -91.60
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 650000
Best mean reward: -91.60 - Last mean reward per episode: -129.62
Num timesteps: 660000
Best mean reward: -91.60 - Last mean reward per episode: -170.40
Num timesteps: 670000
Best mean reward: -91.60 - Last mean reward per episode: -225.91
eps: 11000
(3, 'East', 0)
(4, 'West', 6)
(0, 'Stop', -0.5)
(2, 'Stop', -1)
(3, 'East', -1.5)
(0, 'Stop', 0)
(2, 'Stop', -1)
(0, 'Stop', -1.5)
(4, 'West', -1.5)
(2, 'Stop', 0)
(4, 'West', -1)
(4, 'West', 7)
(4, 'West', 8)
(1, 'North', 9)
(1, 'North', 10)
(4, 'West', 11)
(0, 'Stop', 12)
(0, 'Stop', -7)
(4, 'West', -10.5)
(4, 'West', 13)
(4, 'Stop', 14)
(1, 'North', -9)
(3, 'Stop', 15)
(2, 'South', -10)
(1, 'North', 0)
Num timesteps: 680000
Best mean reward: -91.60 - Last mean reward per episode: -148.43
Num timesteps: 690000
Best mean reward: -91.60 - Last mean reward per episode: -120.94
Num timesteps: 700000
Best mean reward: -91.60 - Last mean reward per episode: -189.65
Num timesteps: 710000
Best mean reward: -91.60 - Last mean reward per episode: -142.62
Num timesteps: 720000
Best mean reward: -91.60 - Last mean reward per episode: -142.69
Num timesteps: 730000
Best mean reward: -91.60 - Last mean reward per episode: -78.77
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 740000
Best mean reward: -78.77 - Last mean reward per episode: -186.94
Num timesteps: 750000
Best mean reward: -78.77 - Last mean reward per episode: -93.25
Num timesteps: 760000
Best mean reward: -78.77 - Last mean reward per episode: -195.42
Num timesteps: 770000
Best mean reward: -78.77 - Last mean reward per episode: -57.08
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 780000
Best mean reward: -57.08 - Last mean reward per episode: -99.09
Num timesteps: 790000
Best mean reward: -57.08 - Last mean reward per episode: -156.40
Num timesteps: 800000
Best mean reward: -57.08 - Last mean reward per episode: -166.51
Num timesteps: 810000
Best mean reward: -57.08 - Last mean reward per episode: -207.24
Num timesteps: 820000
Best mean reward: -57.08 - Last mean reward per episode: -56.48
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 830000
Best mean reward: -56.48 - Last mean reward per episode: -29.39
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 840000
Best mean reward: -29.39 - Last mean reward per episode: -62.95
Num timesteps: 850000
Best mean reward: -29.39 - Last mean reward per episode: -33.63
Num timesteps: 860000
Best mean reward: -29.39 - Last mean reward per episode: -142.27
Num timesteps: 870000
Best mean reward: -29.39 - Last mean reward per episode: -101.22
Num timesteps: 880000
Best mean reward: -29.39 - Last mean reward per episode: -79.75
Num timesteps: 890000
Best mean reward: -29.39 - Last mean reward per episode: -74.66
Num timesteps: 900000
Best mean reward: -29.39 - Last mean reward per episode: -55.02
Num timesteps: 910000
Best mean reward: -29.39 - Last mean reward per episode: -81.06
Num timesteps: 920000
Best mean reward: -29.39 - Last mean reward per episode: -62.77
Num timesteps: 930000
Best mean reward: -29.39 - Last mean reward per episode: -59.62
Num timesteps: 940000
Best mean reward: -29.39 - Last mean reward per episode: -107.15
Num timesteps: 950000
Best mean reward: -29.39 - Last mean reward per episode: -34.71
Num timesteps: 960000
Best mean reward: -29.39 - Last mean reward per episode: 14.28
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 970000
Best mean reward: 14.28 - Last mean reward per episode: 28.89
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 980000
Best mean reward: 28.89 - Last mean reward per episode: 2.45
Num timesteps: 990000
Best mean reward: 28.89 - Last mean reward per episode: 68.64
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1000000
Best mean reward: 68.64 - Last mean reward per episode: 9.56
Num timesteps: 1010000
Best mean reward: 68.64 - Last mean reward per episode: -41.67
Num timesteps: 1020000
Best mean reward: 68.64 - Last mean reward per episode: 21.63
Num timesteps: 1030000
Best mean reward: 68.64 - Last mean reward per episode: 56.90
Num timesteps: 1040000
Best mean reward: 68.64 - Last mean reward per episode: 104.40
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1050000
Best mean reward: 104.40 - Last mean reward per episode: 24.84
Num timesteps: 1060000
Best mean reward: 104.40 - Last mean reward per episode: 103.02
Num timesteps: 1070000
Best mean reward: 104.40 - Last mean reward per episode: 1.80
Num timesteps: 1080000
Best mean reward: 104.40 - Last mean reward per episode: 150.57
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1090000
Best mean reward: 150.57 - Last mean reward per episode: 165.61
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1100000
Best mean reward: 165.61 - Last mean reward per episode: 170.68
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1110000
Best mean reward: 170.68 - Last mean reward per episode: 137.28
Num timesteps: 1120000
Best mean reward: 170.68 - Last mean reward per episode: 94.27
Num timesteps: 1130000
Best mean reward: 170.68 - Last mean reward per episode: 67.98
Num timesteps: 1140000
Best mean reward: 170.68 - Last mean reward per episode: 231.95
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1150000
Best mean reward: 231.95 - Last mean reward per episode: 245.68
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1160000
Best mean reward: 245.68 - Last mean reward per episode: 272.83
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1170000
Best mean reward: 272.83 - Last mean reward per episode: 338.73
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1180000
Best mean reward: 338.73 - Last mean reward per episode: 308.27
Num timesteps: 1190000
Best mean reward: 338.73 - Last mean reward per episode: 274.17
Num timesteps: 1200000
Best mean reward: 338.73 - Last mean reward per episode: 324.94
Num timesteps: 1210000
Best mean reward: 338.73 - Last mean reward per episode: 239.72
Num timesteps: 1220000
Best mean reward: 338.73 - Last mean reward per episode: 295.08
Num timesteps: 1230000
Best mean reward: 338.73 - Last mean reward per episode: 304.70
Num timesteps: 1240000
Best mean reward: 338.73 - Last mean reward per episode: 309.23
Num timesteps: 1250000
Best mean reward: 338.73 - Last mean reward per episode: 259.90
Num timesteps: 1260000
Best mean reward: 338.73 - Last mean reward per episode: 369.93
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1270000
Best mean reward: 369.93 - Last mean reward per episode: 386.48
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1280000
Best mean reward: 386.48 - Last mean reward per episode: 341.04
Num timesteps: 1290000
Best mean reward: 386.48 - Last mean reward per episode: 429.31
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1300000
Best mean reward: 429.31 - Last mean reward per episode: 433.44
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1310000
Best mean reward: 433.44 - Last mean reward per episode: 454.57
Saving new best model to tmp/discrete_pacman_BoardState
eps: 22000
(3, 'East', 0)
(3, 'East', 6)
(3, 'East', 7)
(3, 'East', 8)
(1, 'North', 9)
(1, 'North', 10)
(4, 'West', 11)
(4, 'West', 12)
(4, 'West', 13)
(4, 'West', 14)
(4, 'West', 15)
(4, 'West', 16)
(4, 'West', 17)
(3, 'East', 18)
(4, 'West', -6.5)
(0, 'Stop', -6.5)
(4, 'West', -13)
(4, 'West', 19)
(2, 'South', 20)
(2, 'South', 21)
(4, 'West', 22)
(4, 'West', 23)
(4, 'West', 24)
(1, 'North', 25)
(1, 'North', 26)
(1, 'North', 27)
(1, 'North', 28)
(2, 'South', 29)
(1, 'North', -12.0)
(3, 'East', -12.0)
(3, 'East', 30)
(1, 'North', 31)
(4, 'Stop', 32)
(1, 'North', -27)
(3, 'East', 33)
(4, 'West', 34)
(2, 'South', -14.5)
(1, 'North', 0)
(3, 'East', -14.5)
(3, 'East', 0)
(3, 'East', 35)
(3, 'East', 36)
(3, 'East', 37)
(3, 'East', 38)
(3, 'East', 39)
(3, 'East', 40)
(3, 'East', 41)
(3, 'East', 42)
(3, 'East', 43)
(3, 'East', 44)
(3, 'East', 45)
(4, 'West', 46)
(3, 'East', -20.5)
(2, 'South', -20.5)
(2, 'South', 47)
(4, 'West', 48)
(3, 'East', 49)
(3, 'East', -22.0)
(3, 'East', 50)
(1, 'North', 51)
(1, 'North', 52)
(1, 'North', 53)
(1, 'North', 54)
(2, 'South', 55)
(0, 'Stop', -25.0)
(1, 'North', -50)
(1, 'Stop', 0)
(2, 'South', -50)
(2, 'South', 0)
(2, 'South', 0)
(2, 'South', 0)
(4, 'West', 0)
(4, 'West', 0)
(2, 'South', 0)
(3, 'Stop', 56)
(4, 'Stop', -51)
(1, 'North', -76.5)
Num timesteps: 1320000
Best mean reward: 454.57 - Last mean reward per episode: 429.05
Num timesteps: 1330000
Best mean reward: 454.57 - Last mean reward per episode: 430.77
Num timesteps: 1340000
Best mean reward: 454.57 - Last mean reward per episode: 393.37
Num timesteps: 1350000
Best mean reward: 454.57 - Last mean reward per episode: 402.57
Num timesteps: 1360000
Best mean reward: 454.57 - Last mean reward per episode: 455.19
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1370000
Best mean reward: 455.19 - Last mean reward per episode: 443.50
Num timesteps: 1380000
Best mean reward: 455.19 - Last mean reward per episode: 470.07
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1390000
Best mean reward: 470.07 - Last mean reward per episode: 476.95
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1400000
Best mean reward: 476.95 - Last mean reward per episode: 507.94
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1410000
Best mean reward: 507.94 - Last mean reward per episode: 542.84
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1420000
Best mean reward: 542.84 - Last mean reward per episode: 413.82
Num timesteps: 1430000
Best mean reward: 542.84 - Last mean reward per episode: 422.56
Num timesteps: 1440000
Best mean reward: 542.84 - Last mean reward per episode: 485.45
Num timesteps: 1450000
Best mean reward: 542.84 - Last mean reward per episode: 524.42
Num timesteps: 1460000
Best mean reward: 542.84 - Last mean reward per episode: 446.15
Num timesteps: 1470000
Best mean reward: 542.84 - Last mean reward per episode: 434.15
Num timesteps: 1480000
Best mean reward: 542.84 - Last mean reward per episode: 467.36
Num timesteps: 1490000
Best mean reward: 542.84 - Last mean reward per episode: 371.50
Num timesteps: 1500000
Best mean reward: 542.84 - Last mean reward per episode: 205.58
Num timesteps: 1510000
Best mean reward: 542.84 - Last mean reward per episode: 408.79
Num timesteps: 1520000
Best mean reward: 542.84 - Last mean reward per episode: 536.26
Num timesteps: 1530000
Best mean reward: 542.84 - Last mean reward per episode: 326.51
Num timesteps: 1540000
Best mean reward: 542.84 - Last mean reward per episode: 535.90
Num timesteps: 1550000
Best mean reward: 542.84 - Last mean reward per episode: 637.84
Saving new best model to tmp/discrete_pacman_BoardState
Num timesteps: 1560000
Best mean reward: 637.84 - Last mean reward per episode: 530.41
Num timesteps: 1570000
Best mean reward: 637.84 - Last mean reward per episode: 404.65
Num timesteps: 1580000
Best mean reward: 637.84 - Last mean reward per episode: 329.44
Num timesteps: 1590000
Best mean reward: 637.84 - Last mean reward per episode: 357.51
Num timesteps: 1600000
Best mean reward: 637.84 - Last mean reward per episode: 337.31
Num timesteps: 1610000
Best mean reward: 637.84 - Last mean reward per episode: 403.52
Killed

"""





































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