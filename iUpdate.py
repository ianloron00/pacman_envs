# from Features import *
from os import stat_result
import iFeatures as Features
from gym import spaces
import numpy as np
from game import Directions

class PacmanState:
    
    @staticmethod
    # reset state-related variables
    def reset(pacman):    
        
        extractor = pacman.featExtractor
        
        if isinstance(extractor, Features.Selected):

            # initialize state.
            pacman.frame = PacmanState.get_frame(pacman, pacman.game.state, Directions.STOP)
            pacman.state = np.stack([pacman.frame] * 4)

        else:
            # initialize state.
            pacman.frame = PacmanState.get_frame(pacman, pacman.game.state, Directions.STOP)
            pacman.state = np.stack([pacman.frame] * 4)

            
    @staticmethod
    def get_features(pacman, state, action):
        
        extractor = pacman.featExtractor

        return extractor.getFeatures(pacman, state, action)


    @staticmethod
    def get_frame(pacman, state, action):
        
        extractor = pacman.featExtractor
        
        if isinstance(extractor, Features.Selected):
            pacman.features = PacmanState.get_features(pacman, state, action)

            # get values from dictionary, list them and transform into numpy array.
            return np.array(list(pacman.features.values()))

        else:
            return extractor.getFeatures(pacman, state, action)


    @staticmethod
    def get_state(pacman):
       
        extractor = pacman.featExtractor
        
        if isinstance(extractor, Features.Selected):
            return np.vstack((pacman.state[:-1] , pacman.frame))
        
        else:
            pos = len(pacman.state) - 1
            return np.insert(pacman.state[:-1] , pos,  pacman.frame, axis=0)

    
    @staticmethod
    def initialize_observation_space(pacman):
        
        extractor = pacman.featExtractor
        number_features = extractor.number_features
        

        if isinstance(extractor, Features.Selected):
            # set observation space
            pacman.observation_space = spaces.Box(low=0.0, high=1.0, 
                                    shape=(4, number_features), dtype=np.float16)
        else:
            walls = pacman.game.state.getWalls()
            shape = (4, number_features, walls.width, walls.height)

            pacman.observation_space = spaces.Box(low=-128, high=127, 
                                                  shape=shape, dtype = np.int8)


    @staticmethod
    def update_reward(pacman, new_score, action):
        
        extractor = pacman.featExtractor
        
        if isinstance(extractor, Features.Selected):

            delta = new_score - pacman.last_score
            if delta < 0: delta = 0

            if not pacman.features["pacman-moved"]: 
                delta -= 1
            elif pacman.last_action != None and action == Directions.REVERSE[pacman.last_action]: 
                delta -= 0.2

            if pacman.game.state.isLose(): 
                delta = -30
            elif pacman.game.state.isWin(): 
                delta = 50

            pacman.last_score = new_score
            pacman.reward = delta

        else:
            """
            Increases linearly punctuation to food.
            """
            delta = new_score - pacman.last_score

            # sum of all food - sum number of current food.
            layer_food = pacman.state[-1][2]
            ratio_food = (pacman.n_food - len(layer_food[layer_food != 0]))

            if pacman.game.state.isLose():
                delta = -50 - ratio_food/2
            elif pacman.game.state.isWin():
                delta = 1000
            elif delta < 0:
                delta = 0
            else:
                delta = ratio_food

            if pacman.last_action != None and action == Directions.REVERSE[pacman.last_action]:
                delta = -0.5 * ratio_food
            
            if action == 'Stop':
                delta = -(ratio_food + 1)    

            pacman.last_score = new_score
            pacman.reward = delta