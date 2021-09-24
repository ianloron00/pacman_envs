"""
1- Passar para o Google Colab
2- Treinar mais o pacman

--------------------------
- estado do pacman alterado. 
- Alterado gráfico de curva de aprendizado, para apresentar desvio padrao.
"""

"""
https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

CnnPolicy - para Imagens
MultiInputPolicy - para estados do tipo dicionário

"learning_rate deveria ser alterado? (padrao 0.0001)"
"""

import numpy as np

import gym, os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from iEnvironment import *
from iCallback import *
from iGraphs import *

# 'BoardState' | 'Selected'
extractor = 'BoardState'
name_model = 'discrete_pacman_' + extractor
directory = "tmp/"
isTraining = True

start = 1.0
end = 0.05
fraction = 1.0

env = PacmanEnv.make(extractor=extractor, zoom=2.0)
env = Monitor(env, directory) 

#### applying stable_baselines models.
model = DQN('MlpPolicy', env, tensorboard_log=directory, buffer_size=500000, 
            learning_starts=50000, exploration_fraction=0.5, exploration_initial_eps=1.0, 
            exploration_final_eps=0.05, verbose=0, policy_kwargs=dict( net_arch=[128, 64] ))

def set_expl_variables(model, start=1.0, end=0.05, fraction=1.0):
    model.exploration_fraction = fraction
    model.exploration_initial_eps = start
    model.exploration_final_eps = end
    model.exploration_rate = model.exploration_initial_eps
    model._setup_model()


if not os.path.exists(directory):
    os.mkdir(directory)


if os.path.exists(directory + name_model+".zip"):
    model = model.load(directory + name_model + ".zip", env=env)    
    print("model loaded.")

    set_expl_variables(model, start, end, fraction)


elif os.path.exists(directory + name_model + "/" + name_model+".zip"):
    model = model.load(directory + name_model + "/" + name_model+".zip", env=env)
    print("callback loaded.")

    set_expl_variables(model, start, end, fraction)


if isTraining:
    timesteps = 15e6

    print("initial: {}, final: {}, frac: {}, progr: {}".format(
                        model.exploration_initial_eps, 
                        model.exploration_final_eps, 
                        model.exploration_fraction,
                        model._current_progress_remaining))

    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=directory, name=name_model)
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # plot rewards of training
    plt_training([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Pacman", name_model)

    # double-save
    model.save(directory)
    print("model saved.")

else:
    rewards = evaluate_policy(model, env, n_eval_episodes=100, render=False, return_episode_rewards=True)

    # plot results from policy evaluation
    scores=rewards[0]

    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(scores)) + 1, scores)
    plt.title("Pacman reward evaluation | DQN")
    plt.savefig(directory + "Pacman R eval. | extractor " + extractor + " | DQN" + '.png')
    plt.show()
    plt.close()

env.close()