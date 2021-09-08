# Written by Ian Loron de Almeida.

from numpy.lib.utils import info
# from AgentFiles.FeaturesPolicy import Extractor
from FeaturesEnvSpace import Extractor
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
        self.observation_space=None
        self.rules = None
        self.reward = 0
        self.last_score = 0
        self.shape = None
        self.info = {}
        self.features = None
        self.state = np.array([])
        self.featExtractor = util.lookup('Extractor', globals())()
        self.converted_action_space = [Directions.STOP, Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        """
        # Feature Extractor has the following features:
        # bias, n-[scared]-1-step, n-2-steps, n-[active]-1-step, n-2-steps, # (eats-food),
        # (closest-food)
        """

    def make(**args):
        # create pacman environment
        pacman = PacmanEnv()

        # initialize a game
        pacman.initialize(**args)
        return pacman
    

    def reset(self):
        # reset reward variables
        self.reward = 0 
        self.last_score = 0
        
        # create game        
        self.game = self.createGame(**self.args.copy())

        # should return state.
        self.state = self.get_features(Directions.STOP)
        return self.state


    def render(self, mode='human'):
        self.game.render = True

    
    def step(self, action):

        # convert action to Discrete.
        action = self.valueToAction(action)

        self.game.step(action=action)

        # update features
        self.features = self.get_features(action)

        #update state
        # np.append(self.state, self.features)
        # if self.state.size > 5 : self.state = self.state[1:]
        self.state = self.features

        # update reward
        self.reward = self.game.state.data.score - self.last_score
        self.last_score = self.game.state.data.score

        return (self.state, self.reward, self.game.gameOver, self.info)
    

    def close(self):
        self.game.display.finish() 


    def get_features(self, action):
        # extract features
        features = self.featExtractor.getFeatures(self.game.state, action)

        # get values from dictionary, list them and transform into numpy array.
        return np.array(list(features.values()))


    def validade_action(self, action):
        if action not in self.game.state.getLegalPacmanActions():
            return Directions.STOP
        return action


    def ActionToValue(self, action):
        if type(action) == type(Directions.STOP):
            return np.where(action == self.converted_action_space)


    def valueToAction(self, action):
        # actions can be either numbers or Directions.
        if type(action) != type(Directions.STOP):
            action = self.converted_action_space[action]

        return self.validade_action(action)


    """
    'initialize' takes all parameters and convert to a string, to reuse former pacman functions. 
    """
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

        self.initialize_action_space()
        self.initialize_observation_space()

    def initialize_action_space(self):      
        # set action space
        self.action_space = Discrete(5)

    def initialize_observation_space(self):
        obs_shape_high = np.array([0.1] + [0.2]*4  + [0.1]*(Extractor.number_features - 5))
        obs_shape_low  = np.array([0.1] + [0.0]*(Extractor.number_features-1))

        # set observation space
        self.observation_space = spaces.Box(low=obs_shape_low, high=obs_shape_high, 
                                 shape=obs_shape_high.shape, dtype=np.float64)


    def createGame(self, layout, pacman, ghosts, display, numGames, record=False, numTraining = 0, catchExceptions=False, timeout=30 ):
        import __main__
        __main__.__dict__['_display'] = display

        rules = ClassicGameRules(timeout)
        return rules.newGame(layout, pacman, ghosts, display)


    # returns random Direction.
    def getRandomAction(self):
        legal = self.game.state.getLegalPacmanActions()
        return np.random.choice(legal) if legal != [] else None


    def _save_obs(self):
        return self.observation_space