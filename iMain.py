"""
1- Passar para o Google Colab
2- Treinar mais o pacman

--------------------------
- estado do pacman alterado. 
- Alterado gráfico de curva de aprendizado, para apresentar desvio padrao.
"""

"""
https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
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

if not os.path.exists(directory):
    os.mkdir(directory)

# True - trains de agent. False - evaluates the NN.
isTraining = True

env = PacmanEnv.make(extractor=extractor, zoom=2.0)
env = Monitor(env, directory) 

#### applying stable_baselines3 model.
model = DQN('MlpPolicy', env, tensorboard_log=directory, buffer_size=200_000, 
            learning_starts=50_000, exploration_fraction=0.5, exploration_initial_eps=1.0, 
            exploration_final_eps=0.05, verbose=0, policy_kwargs=dict( net_arch=[128, 64] ))

def get_expl_variables(model):
    start = model.exploration_rate
    end = model.exploration_final_eps
    fraction = max(0, (model.exploration_rate - model.exploration_final_eps)/(model.exploration_initial_eps - model.exploration_final_eps) * model.exploration_fraction)
    return start, end, fraction

def set_expl_variables(model, start=1.0, end=0.05, fraction=1.0):
    model.exploration_initial_eps = start
    model.exploration_final_eps = end
    model.exploration_fraction = fraction
    model._setup_model()

def load_model(model, zip_dir, buffer_dir, env):
    model = model.load(zip_dir, env=env)
    
    model.load_replay_buffer(buffer_dir + "/replay_buffer")

    start, end, fraction = get_expl_variables(model)
    set_expl_variables(model, start, end, fraction)
    return model

if os.path.exists(directory + name_model+".zip"):
    model = load_model(model, directory + name_model + ".zip", directory, env)
    print("model loaded.")

elif os.path.exists(directory + name_model + "/" + name_model + ".zip"):
    model = load_model(model, directory + name_model + "/" + name_model + ".zip", directory, env=env)
    print("callback loaded.")

if isTraining:
    timesteps = 15e6

    print("initial: {}, final: {}, frac: {}, remaining: {}".format(
                        model.exploration_initial_eps, 
                        model.exploration_final_eps, 
                        model.exploration_fraction,
                        model._current_progress_remaining))

    callback = SaveOnBestTrainingRewardCallback(check_freq=100_000, log_dir=directory, name=name_model)
    model.learn(total_timesteps=timesteps, callback=callback)
    
    # plot rewards of training
    plt_training([directory], timesteps, results_plotter.X_TIMESTEPS, "DQN Pacman", name_model)

    # double-save
    model.save(directory)
    print("model saved.")

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