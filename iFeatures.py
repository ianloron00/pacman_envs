from game import Directions, Actions
import util
from BFSPolicy import *
import numpy as np

class Selected:
    name_features = np.array(["bias", "#-of-scared-ghosts-1-step-away", "#-of-scared-ghosts-2-steps-away",
                    "#-of-active-ghosts-1-step-away", "#-of-active-ghosts-2-steps-away", "closest-food",
                    "eats-food", "run-to-catch-closest-food", "is-possible-action", "is-not-a-wall",
                    "has-not-a-wall-north","has-not-a-wall-south","has-not-a-wall-east",
                    "has-not-a-wall-west", "eats-capsule", "pacman-moved", "last-action", "x_pos", "y_pos", 
                    "min-dist-scared-ghosts", "min-dist-active-ghosts", "dir-to-closest-scared-ghost", 
                    "dir-to-closest-active-ghost"])

    number_features = name_features.size

    def convert_action_to_value(self, action, possible_actions):
        return np.where(str(action) == possible_actions)[0][0] / 4.0 if action != None else 0.0

    def getFeatures(self, pacman, state, action):
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

        n_ghosts = len(ghosts)

        # finds if there are close scared ghosts: 
        features["#-of-scared-ghosts-1-step-away"] = (sum((next_x, next_y) in 
                Actions.getLegalNeighbors(ghosts[g], walls) for g in range(len(ghosts)) if 
                ghostScaredTime(g+1,state) > 0) / n_ghosts
        )
        features["#-of-scared-ghosts-2-steps-away"] = (sum(a in 
                Actions.getLegalNeighbors(ghosts[g], walls) for a in next_actions 
                for g in range(len(ghosts)) if ghostScaredTime(g+1,state) > 0) / (4 * n_ghosts)
        )
        # finds if there is active ghost nearby:
        features["#-of-active-ghosts-1-step-away"] = (sum((next_x, next_y) in 
                Actions.getLegalNeighbors(ghosts[g], walls) for g in range(len(ghosts)) if 
                ghostScaredTime(g + 1,state) <= 0) / n_ghosts
        )
        features["#-of-active-ghosts-2-steps-away"] = (sum(a in 
                Actions.getLegalNeighbors(ghosts[g], walls) for a in next_actions for g in 
                range(len(ghosts)) if ghostScaredTime(g+1,state) <= 0) / (4 * n_ghosts)
        )

        fruit = Graph.getClosestPos((next_x, next_y), walls, food)
        
        # if fruit is not None:
        features["closest-food"] = float(fruit.dist) / (walls.width * walls.height) if fruit != None else 0.0
        features["eats-food"] = 1.0 if (not features["#-of-active-ghosts-1-step-away"] and food[next_x][next_y]) else 0.0
        features["run-to-catch-closest-food"] = float(int(action==fruit.dir)/(fruit.dist+1)) if fruit != None else 0.0
        
        legal = state.getLegalPacmanActions()
        features["is-possible-action"] = float(action in legal[:-1])

        # evaluate if the next position is a good choice.
        x_wall = next_x
        y_wall = walls.height - 1 - next_y

        features['is-not-a-wall'] = float(not walls[x_wall][y_wall])
        features["has-not-a-wall-north"] = float(not walls[x_wall][max(0, y_wall - 1)])
        features["has-not-a-wall-south"] = float(not walls[x_wall][min(walls.height - 1, y_wall + 1)])
        features["has-not-a-wall-east"] = float(not walls[max(0, x_wall - 1)][y_wall])
        features["has-not-a-wall-west"] = float(not walls[min(walls.width - 1, x_wall + 1)][y_wall])
        
        # may incentive the agent to eat capsules.
        features["eats-capsule"] = float((next_x,next_y) in state.getCapsules()) if not features["#-of-active-ghosts-1-step-away"] else 0.0

        features["pacman-moved"] =  float((next_x, next_y) != (x, y))  
        
        # (0 - stop, 1 - north, 2 - south, 3 - east, 4 - west) / 4.0
        last_action = pacman.last_action
        features["last-action"] = self.convert_action_to_value(last_action, pacman.POSSIBLE_ACTIONS)
        
        # pacman's position
        features["x_pos"] = x / (walls.width - 1.0)
        features["y_pos"] = y / (walls.height - 1.0)
        
        # compute ghosts-related features
        scared_ghosts = [ghosts[s] for s in range(len(ghosts)) if ghostScaredTime(s+1,state) > 0]
        active_ghosts = [g for g in ghosts if g not in scared_ghosts]

        scared = Graph.getClosestPos((next_x, next_y), walls, scared_ghosts)
        active = Graph.getClosestPos((next_x, next_y), walls, active_ghosts)

        features["min-dist-scared-ghosts"] = scared.dist / (2*(walls.height + walls.width)) if scared != None else 1.0
        features["min-dist-active-ghosts"] = active.dist / (2*(walls.height + walls.width)) if active != None else 1.0
        
        features["dir-to-closest-scared-ghost"] = self.convert_action_to_value(scared.dir, pacman.POSSIBLE_ACTIONS) if scared != None else 0.0 
        features["dir-to-closest-active-ghost"] = self.convert_action_to_value(active.dir, pacman.POSSIBLE_ACTIONS) if active != None else 0.0

        return features

def ghostScaredTime(index, state):
    return state.getGhostState(index).scaredTimer

class BoardState:
    
    # number of layers
    number_features = 7

    def __init__(self):
        self.layer_visited = []     

    def getFeatures(self, pacman, state, action):
        walls = state.getWalls()
        food = state.getFood()
        ghosts = state.getGhostPositions()
        capsules = state.getCapsules()
        
        x, y = state.getPacmanPosition()

        layer_walls = 127 * np.array(walls.data).astype(np.int8)
        
        layer_food = 127 * np.array(food.data).astype(np.int8)
        
        layer_pacman = np.zeros(shape=layer_walls.shape, dtype=np.int8)
        layer_pacman[x,y] = 127

        if not pacman.timestep:
            self.layer_visited = layer_pacman.copy()
        else:
            self.layer_visited[x,y] = 127 

        layer_ghosts = np.zeros(shape=layer_walls.shape, dtype=np.int8)

        for n in range(len(ghosts)):
            g_x, g_y = (int(i) for i in ghosts[n])
            val = 127 if ghostScaredTime(n + 1,state) > 0 else -128
            layer_ghosts[g_x][g_y] = val
        
        layer_caps = np.zeros(shape=layer_walls.shape, dtype=np.int8)
        
        for c in capsules:
            layer_caps[int(c[0])][int(c[1])] = 127 

        layer_data = np.zeros(shape=layer_walls.shape, dtype=np.int8)

        layer_data[0][0] = len(layer_food[layer_food != 0])
        for g in range(len(ghosts)):
            layer_data[1][g] = ghostScaredTime(g + 1, state)

        features = np.concatenate((self.layer_visited, layer_walls, layer_food, layer_pacman, 
                                   layer_ghosts, layer_caps, layer_data ))

        features = np.array(np.array_split(features, self.number_features))
        
        return features
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