# Written by Ian Loron de Almeida.

from AgentFiles.FeaturesPolicy import Extractor
import math, util
from tokenize import String
import gym
from gym import spaces
from gym.spaces.discrete import Discrete
from gym.utils import seeding
import numpy as np

from pacman import *
from game import *

# Create an environment that is equivalent to Gym's standard environments.
# It should contain:
"""
.render()
.step(action)
.reset()
.close()
.seed(seed=None)

get_keys_to_action(self):
clone_state(self):
restore_state(self, state):
clone_full_state(self):
restore_full_state(self, state):
get_action_meanings(self):

_get_image(self):
_n_actions(self):
_get_obs(self):
"""
"""
.observation_space()
.action_space()
.action_space.sample()
.goal_position() # numFood=0
"""

class PacmanEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, maze_name='mediumClassic'):
        super(PacmanEnv, self).__init__()
        
        self.args = ""
        self.game = None
        self.action_space=None
        self.rules = None
        self.reward = 0
        self.last_score = 0
        self.shape = None
        self.featExtractor = util.lookup('Extractor', globals())()

        self.action_space = np.arange(0,5) # Discrete(5)
        self.converted_action_space = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        """
        # Feature Extractor has the following features:
        # bias, n-[scared]-1-step, n-2-steps, n-[active]-1-step, n-2-steps, # (eats-food),
        # (closest-food), scaredTime, run-to-catch
        """
        n_ghosts = 2 # self.game.state.data.getNumAgents() - 1
        obs_shape = np.array((1.0, n_ghosts, n_ghosts, n_ghosts, n_ghosts, 1.0, 1.0, 1.0, 1.0))        
        self.observation_space = spaces.Box(low=[0]*obs_shape, high=obs_shape, 
                                 shape=obs_shape.shape, dtype=np.float64) 

    def make(**args):
        pacman = PacmanEnv()
        pacman.initialize(**args)
        return pacman
    
    def reset(self): 
        self.reward = 0 
        self.last_score = 0
        self.game = self.createGame(**self.args.copy())
        # self.action_space = self.game.data.getLegalPacmanActions()
        return self.observation_space.low # ??

    def render(self):
        self.game.render = True
        return

    def update_obs_space(self, state, action):
        features = self.featExtractor.getFeatures(self.game.state, action)

        # get values from dictionary, list them and transform into numpy array.
        print("#####################")
        print(list(features.values()))
        return np.array(list(features.values()))

    def update_action_space(self):
        legal_actions = self.game.state.getLegalPacmanActions()
        all_actions = self.converted_action_space
        return [i for i in range(len(all_actions)) for j in legal_actions if all_actions[i] == j]
    
    def valueToAction(self, action):
        return self.converted_action_space[action]

    def step(self, action):
        state = self.game.state

        # convert action to Discrete.
        action = self.valueToAction(action)

        self.game.step(action=action)

        # update reward
        self.reward = self.game.state.data.score - self.last_score
        self.last_score = self.game.state.data.score

        # redefine observation_space
        self.observation_space = self.update_obs_space(state, action)

        # update action space
        self.action_space = self.update_action_space()

        # return all information needed        
        return (self.observation_space, self.reward, self.game.gameOver, "dont know what to say")
    
    def close(self):
        self.game.display.finish() 

    def initialize(self, layout=None, numTraining=None, numGames=None, numGhosts=None, zoom=None, display=2):
        myArgs=[]
        if layout: myArgs.append("-l"+layout)
        if numTraining: myArgs.append("-x "+str(numTraining))
        if numGames: myArgs.append("-n "+str(numGames))
        if numGhosts: myArgs.append("-k "+str(numGhosts))
        if zoom: myArgs.append("-z "+str(zoom))
        if display == 0: myArgs.append('-q')
        elif display == 1: myArgs.append('-t')

        self.args = readCommand(myArgs)
        self.game = self.createGame(**self.args.copy())

        self.action_space = self.update_action_space()
        # self.observation_space = self.update_obs_space()

    def createGame(self, layout, pacman, ghosts, display, numGames, record=False, numTraining = 0, catchExceptions=False, timeout=30 ):
        import __main__
        __main__.__dict__['_display'] = display

        rules = ClassicGameRules(timeout)
        return rules.newGame(layout, pacman, ghosts, display)

    def getRandomAction(self):
        legal = self.game.state.getLegalPacmanActions()
        return np.random.choice(legal) if legal != [] else None

    def _save_obs(self):
        return self.observation_space

# temporary
# def __getInput__(legal):  
#     cin = str(input())
#     action = Directions.STOP
#     if cin == 'a' and Directions.WEST in legal:
#         action = Directions.WEST
#     elif cin == 'd' and Directions.EAST in legal:
#         action = Directions.EAST
#     elif cin == 'w' and Directions.NORTH in legal:
#         action = Directions.NORTH
#     elif cin == 's' and Directions.SOUTH in legal:
#         action = Directions.SOUTH
#     return action