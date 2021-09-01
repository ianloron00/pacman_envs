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
        self.featExtractor = util.lookup('Extractor', globals())()
        self.converted_action_space = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        """
        # Feature Extractor has the following features:
        # bias, n-[scared]-1-step, n-2-steps, n-[active]-1-step, n-2-steps, # (eats-food),
        # (closest-food), scaredTime, run-to-catch
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

        # return possible pacman movements
        return self.observation_space.low

    def render(self, mode='human'):
        self.game.render = True

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
        if type(action) == type(Directions):
            return np.where(action == self.converted_action_space)

    def valueToAction(self, action):
        if type(action)==type(1):
            action = self.converted_action_space[action]
        return self.validade_action(action)

    def step(self, action):
        # convert action to Discrete.
        action = self.valueToAction(action)

        self.game.step(action=action)

        # update state
        self.features = self.get_features(action)

        # update reward
        self.reward = self.game.state.data.score - self.last_score
        self.last_score = self.game.state.data.score

        # return all information needed        
        return (self.features, self.reward, self.game.gameOver, self.info)
    
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

        # walls = self.game.state.data.layout.walls
        # self.action_space = Discrete(5*(walls.width*walls.height))
        
        # set action space
        self.action_space = Discrete(5)

        n_ghosts = self.game.state.getNumAgents() - 1
        obs_shape = np.array((1.0, n_ghosts, n_ghosts, n_ghosts, n_ghosts, 1.0, 1.0)) #, (0))) # , 1.0, 1.0))        

        # set observation space        
        self.observation_space = spaces.Box(low=[0]*obs_shape, high=obs_shape, 
                                 shape=obs_shape.shape, dtype=np.float64) 

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