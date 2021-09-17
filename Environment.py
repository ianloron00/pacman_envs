# Written by Ian Loron de Almeida.

from os import stat_result
from gym.core import RewardWrapper
from numpy.lib.utils import info

import math, util
from tokenize import String
import gym
from gym import spaces
from gym.spaces.discrete import Discrete
from gym.utils import seeding
import numpy as np

from pacman import *
from game import *
from State import PacmanState as State
from Features import *
 
# Create an environment that is equivalent to Gym's standard environments.
# It should contain:
"""
.render()
.step(action)
.reset()
.close()
.seed(seed=None)
"""

class PacmanEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    POSSIBLE_ACTIONS = np.array([Directions.STOP, Directions.NORTH, Directions.SOUTH, 
                                Directions.EAST, Directions.WEST])

    def __init__(self, maze_name='mediumClassic',extractor='BoardState'):
        
        super(PacmanEnv, self).__init__()
        
        self.args = ""
        self.game = None
        self.rules = None

        self.action_space = None
        self.observation_space = None
        self.converted_action_space = self.POSSIBLE_ACTIONS

        self.reward = 0
        self.last_score = 0
        self.last_action = None
        self.info = {}

        self.featExtractor = util.lookup(extractor, globals())()
        self.features = np.array([])
        self.frame = np.array([])
        self.state = np.array([])

        self.eps = 0


    def make(extractor='BoardState', **args):
        # create pacman environment
        print(extractor)
        pacman = PacmanEnv(extractor = extractor)

        # initialize a game
        pacman.initialize(**args)
        return pacman
    

    def reset(self):
        self.eps += 1
        if not (self.eps % 200): print("eps: {}".format(self.eps))
        
        # reset reward variables
        self.reward = 0
        self.last_score = 0

        # reset last action
        self.last_action = None
        
        # create game        
        self.game = self.createGame(**self.args.copy())

        # reset state-related variables
        State.reset(self)

        # should return state.
        return self.state


    def render(self, mode='human'):
        self.game.render = True


    def step(self, action):
        if not (self.eps % 200):
            print(str((action, self.valueToAction(action), self.reward)))

        # print(str((action, self.valueToAction(action), self.reward)))

        # convert action to Discrete.
        action = self.valueToAction(action)

        self.game.step(action=action)

        # update features
        self.frame = State.get_frame(self, self.game.state, action)

        # update state
        self.state = State.get_state(self)
        
        # update reward
        State.update_reward(self, self.game.state.data.score, action)

        # update last action
        self.last_action = action

        return (self.state, self.reward, self.game.gameOver, self.info)
    

    def close(self):
        self.game.display.finish() 


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
    def initialize(self, extractor='BoardState', layout=None, numTraining=None, numGames=None, numGhosts=None, zoom=None, display=2):
        
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
        State.initialize_observation_space(self)


    def createGame(self, layout, pacman, ghosts, display, numGames, record=False, numTraining = 0, catchExceptions=False, timeout=30 ):
        import __main__
        __main__.__dict__['_display'] = display

        rules = ClassicGameRules(timeout)
        return rules.newGame(layout, pacman, ghosts, display)


    # returns random Direction.
    def getRandomAction(self):
        legal = self.game.state.getLegalPacmanActions()
        return np.random.choice(legal) if legal != [] else None