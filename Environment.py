# Written by Ian Loron de Almeida.

import math
# from pickle import READONLY_BUFFER
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
    # .start()
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

    def __init__(self, maze_name='mediumClassic', enable_render=False):
        super(PacmanEnv, self).__init__()
        
        self.enable_render=enable_render
        self.maze_name=maze_name
        self.env=None
        self.args=""
        self.game=None
        self.zoom=1.0
        self.frameTime=0.0
        self.timeout=30
        self.textGraphics=False
        self.quietGraphics=False
        self.action_space=None
        self.rules = None
        self.layout = None
        self.pacman = None
        self.ghosts = None
        self.display = None
        self.reward = 0
        self.last_score = 0  

    def make(**args):
        pacman = PacmanEnv()
        pacman.initialize(**args)
        return pacman
    
    def reset(self): 
        self.reward = 0 
        self.last_score = 0
        self.game = self.createGame(**self.args.copy())
        return self.game.state

    def render(self):
        self.game.render = True
        return

    def step(self, action):
        self.game.step(action=action)
        self.reward = self.game.state.data.score - self.last_score
        self.last_score = self.game.state.data.score
        return (self.game.state, self.reward, self.game.gameOver, "dont know what to say")
    
    def close(self):
        self.game.display.finish()
        return 

    def initialize(self, layout=None, numTraining=None, numGames=None, numGhosts=None, zoom=None, display=2):
        myArgs=[]
        if layout: myArgs.append("-l"+layout)
        if numTraining: myArgs.append("-x "+str(numTraining))
        if numGames: myArgs.append("-n "+str(numGames))
        if numGhosts: myArgs.append("-k "+str(numGhosts))
        if zoom: myArgs.append("-z "+str(zoom))
        if display==0: myArgs.append('-q')
        elif display==1: myArgs.append('-t')

        self.args = readCommand(myArgs)
        self.game = self.createGame(**self.args.copy())
    
    def getRandomAction(self):
        legal = self.game.state.getLegalPacmanActions()
        return np.random.choice(legal) if legal != [] else None

    def createGame(self, layout, pacman, ghosts, display, numGames, record=False, numTraining = 0, catchExceptions=False, timeout=30 ):
        import __main__
        __main__.__dict__['_display'] = display

        rules = ClassicGameRules(timeout)
        return rules.newGame(layout, pacman, ghosts, display)

# temporary
def __getInput__(legal):  
    cin = str(input())
    action = Directions.STOP
    if cin == 'a' and Directions.WEST in legal:
        action = Directions.WEST
    elif cin == 'd' and Directions.EAST in legal:
        action = Directions.EAST
    elif cin == 'w' and Directions.NORTH in legal:
        action = Directions.NORTH
    elif cin == 's' and Directions.SOUTH in legal:
        action = Directions.SOUTH
    return action