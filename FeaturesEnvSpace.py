from game import Directions, Actions
import util
from BFSPolicy import *
import numpy as np

class Extractor:
    converted_action_space = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]

    def getFeatures(self,state,action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # compute legal forward actions
        next_actions = Actions.getLegalNeighbors((next_x,next_y), walls)

        # finds if there is closed scared ghosts: 
        features["#-of-scared-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(ghosts[g], walls) for g in range(len(ghosts)) if ghostScaredTime(g+1,state) >0)
        features["#-of-scared-ghosts-2-steps-away"] = sum(a in Actions.getLegalNeighbors(ghosts[g], walls) for a in next_actions for g in range(len(ghosts)) if ghostScaredTime(g+1,state) >0)

        # finds if there is activ ghost nearby:
        features["#-of-active-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(ghosts[g], walls) for g in range(len(ghosts)) if ghostScaredTime(g+1,state) <=0)
        features["#-of-active-ghosts-2-steps-away"] = sum(a in Actions.getLegalNeighbors(ghosts[g], walls) for a in next_actions for g in range(len(ghosts)) if ghostScaredTime(g+1,state) <=0)

        # if not features["#-of-active-ghosts-1-step-away"] and bool((next_x,next_y) in state.getCapsules()):
        #     features['has-capsule-1-step-away']=1.0
        
        # # if not features["#-of-active-ghosts-1-step-away"] and food[next_x][next_y]:

        fruit = Graph.getClosestPos((next_x, next_y), walls, food)
        
        # if fruit is not None:
        dist=fruit.dist
        features["closest-food"] = float(dist) / (walls.width * walls.height)
            # dir=fruit.dir
            # features["run-to-catch-closest-fruit"]=float(int(action==dir)/(dist+1))

        # scared_ghosts=[ghosts[s] for s in range(len(ghosts)) if ghostScaredTime(s+1,state)>0]
        # scared_ghost=Graph.getClosestPos((next_x, next_y), walls, scared_ghosts)
        
        # try to run to catch scared ghosts.
        # if scared_ghost != None:
            # scared_ghost_dist=scared_ghost.dist
            # scared_ghost_index=scared_ghost.ghost_id 
            # scared_ghost_dir=scared_ghost.dir
# 
            # features["scaredTime-ghostDist"] = float(ghostScaredTime(scared_ghost_index+1,state))*0.5 - float(scared_ghost_dist) / (walls.width * walls.height)
            # features["run-to-catch-scared-ghost"]=float(int(action == scared_ghost_dir)/(
                # scared_ghost_dist+1))*0.5

        features["eats-food"] = 1.0 if (not features["#-of-active-ghosts-1-step-away"] and food[next_x][next_y]) else 0

        """
        change observation space interval too in order to use this feature.
        """
        # legal = state.getLegalPacmanActions()
        # actions = Extractor.converted_action_space
        # features["possible-actions"] = np.array([x for x in range(len(actions)) if actions[x] in legal])

        features.divideAll(10.0)

        return features

def ghostScaredTime(index, state):
    return state.getGhostState(index).scaredTimer