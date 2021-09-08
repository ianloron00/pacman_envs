from game import Directions, Actions
import util
from BFSPolicy import *
import numpy as np

class Extractor:
    # converted_action_space = [Directions.STOP, Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
    number_features = len(["bias", "#-of-scared-ghosts-1-step-away", "#-of-scared-ghosts-2-steps-away",
                    "#-of-active-ghosts-1-step-away", "#-of-active-ghosts-2-steps-away", "closest-food",
                    "eats-food", "run-to-catch-closest-fruit", "is-possible-action", "is-not-a-wall",
                    "has-not-a-wall-north","has-not-a-wall-south","has-not-a-wall-east",
                    "has-not-a-wall-west", "eats-capsule"])

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

        fruit = Graph.getClosestPos((next_x, next_y), walls, food)
        
        # if fruit is not None:
        features["closest-food"] = float(fruit.dist) / (walls.width * walls.height) if fruit != None else 0
        features["eats-food"] = 1.0 if (not features["#-of-active-ghosts-1-step-away"] and food[next_x][next_y]) else 0
        features["run-to-catch-closest-fruit"]=float(int(action==fruit.dir)/(fruit.dist+1)) if fruit != None else 0
        
        legal = state.getLegalPacmanActions()
        features["is-possible-action"] = int(action in legal[0:-1])

        # evaluate if the next position is a good choice.
        x_wall = next_x
        y_wall = walls.height - 1 - next_y

        features['is-not-a-wall'] = float(not walls[x_wall][y_wall])
        features["has-not-a-wall-north"] = float(not walls[x_wall][min(walls.height - 1, y_wall + 1)])
        features["has-not-a-wall-south"] = float(not walls[x_wall][max(0, y_wall - 1)])
        features["has-not-a-wall-east"] = float(not walls[max(0, x_wall - 1)][y_wall])
        features["has-not-a-wall-west"] = float(not walls[min(walls.width - 1, x_wall + 1)][y_wall])
        
        # may incentive the agent to eat capsules.
        features["eats-capsule"] = float((next_x,next_y) in state.getCapsules()) if not features["#-of-active-ghosts-1-step-away"] else 0


        features.divideAll(10.0)
        return features

def ghostScaredTime(index, state):
    return state.getGhostState(index).scaredTimer

"""
walls:
.---------> x
|
|
|
v

y


pacman:

y

^
|
|
|
.--------> x
"""