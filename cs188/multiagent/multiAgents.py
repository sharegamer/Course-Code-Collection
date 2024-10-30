# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        nearest_ghost_dis = 1e9
        for ghost_state in newGhostStates:
            ghost_x, ghost_y = ghost_state.getPosition()
            ghost_x = int(ghost_x)
            ghost_y = int(ghost_y)
            if ghost_state.scaredTimer == 0:
                nearest_ghost_dis = min(nearest_ghost_dis,\
                                        manhattanDistance((ghost_x, ghost_y), newPos))
        
        
        if nearest_ghost_dis<2:
            nearest_ghost_dis=-1000
        else:
            nearest_ghost_dis=0
        
        food = currentGameState.getFood()
        if food[newPos[0]][newPos[1]]:
            minFoodDist = 0
        else:
            foodDistances = [
                manhattanDistance(newPos, (x, y))
                for x in range(food.width)
                for y in range(food.height)
                if food[x][y]
            ]
            minFoodDist = min(foodDistances, default=0)
        return currentGameState.getScore()+nearest_ghost_dis-minFoodDist

       

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
   
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    nearest_ghost_dis = 1e9
    for ghost_state in newGhostStates:
        ghost_x, ghost_y = ghost_state.getPosition()
        ghost_x = int(ghost_x)
        ghost_y = int(ghost_y)
        if ghost_state.scaredTimer == 0:
            nearest_ghost_dis = min(nearest_ghost_dis,\
                                    manhattanDistance((ghost_x, ghost_y), newPos))
    
    
    food = currentGameState.getFood()
    dis=0
    foodDistances = [
            manhattanDistance(newPos, (x, y))
            for x in range(food.width)
            for y in range(food.height)
            if food[x][y]
    ]
    dis = sum(foodDistances)
    return currentGameState.getScore()+nearest_ghost_dis-dis


    # return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        value,action=self.maxAction(self.depth,gameState)
        return action
        

    # first return is max value second return is correspond action
    def maxAction(self,depth:int,gameState:GameState):
        if gameState.isWin() or gameState.isLose():
            return scoreEvaluationFunction(gameState),None
        if depth==0:
            return scoreEvaluationFunction(gameState),None
        actions=gameState.getLegalActions(0)
        maxValue=-float('inf')
        for action in actions:
            if maxValue<self.minAction(depth,gameState.generateSuccessor(0,action),1):
                maxValue=max(maxValue,self.minAction(depth,gameState.generateSuccessor(0,action),1))
                max_action=action
        return maxValue,max_action
        
            
    
    def minAction(self,depth:int ,gameState:GameState,agentIndex:int)->float:        
            if gameState.isWin() or gameState.isLose():
                return scoreEvaluationFunction(gameState)
            if depth==0:
                return scoreEvaluationFunction(gameState)
            actions=gameState.getLegalActions(agentIndex)
            minValue=float('inf')
            for action in actions:
                if agentIndex==gameState.getNumAgents()-1:
                    tmp1,tmp2=self.maxAction(depth-1,gameState.generateSuccessor(agentIndex,action))
                    minValue=min(minValue,tmp1)
                else:
                    minValue=min(minValue,self.minAction(depth,gameState.generateSuccessor(agentIndex,action),agentIndex+1))
            return minValue
            
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value,action=self.maxAction(self.depth,gameState,-float('inf'),float('inf'))
        return action
        
        
    def maxAction(self,depth:int,gameState:GameState,alpha:float,beta:float):
        if gameState.isWin() or gameState.isLose():
            return scoreEvaluationFunction(gameState),None
        if depth==0:
            return scoreEvaluationFunction(gameState),None
        actions=gameState.getLegalActions(0)
        maxValue=-float('inf')
        for action in actions:
            value=self.minAction(depth,gameState.generateSuccessor(0,action),1,alpha,beta)
            if maxValue<value:
                maxValue=value
                max_action=action
            if maxValue>beta:
                return maxValue,max_action
            if maxValue>alpha:
                alpha=maxValue
        return maxValue,max_action
    
    
    def minAction(self,depth:int ,gameState:GameState,agentIndex:int,alpha:float,beta:float)->float:        
            if gameState.isWin() or gameState.isLose():
                return scoreEvaluationFunction(gameState)
            if depth==0:
                return scoreEvaluationFunction(gameState)
            actions=gameState.getLegalActions(agentIndex)
            minValue=float('inf')
            for action in actions:
                if agentIndex==gameState.getNumAgents()-1:
                    tmp1,tmp2=self.maxAction(depth-1,gameState.generateSuccessor(agentIndex,action),alpha,beta)
                    minValue=min(minValue,tmp1)
                else:
                    minValue=min(minValue,self.minAction(depth,gameState.generateSuccessor(agentIndex,action),agentIndex+1,alpha,beta))
                if minValue<alpha:
                    return minValue
                if minValue<beta:
                    beta=minValue
            return minValue
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        value,action=self.maxAction(self.depth,gameState)
        return action

    def maxAction(self,depth:int,gameState:GameState):
        if gameState.isWin() or gameState.isLose():
            return scoreEvaluationFunction(gameState),None
        if depth==0:
            return scoreEvaluationFunction(gameState),None
        actions=gameState.getLegalActions(0)
        maxValue=-float('inf')
        for action in actions:
            if maxValue<self.minAction(depth,gameState.generateSuccessor(0,action),1):
                maxValue=max(maxValue,self.minAction(depth,gameState.generateSuccessor(0,action),1))
                max_action=action
        return maxValue,max_action
    
    def minAction(self,depth:int ,gameState:GameState,agentIndex:int)->float:        
            if gameState.isWin() or gameState.isLose():
                return scoreEvaluationFunction(gameState)
            if depth==0:
                return scoreEvaluationFunction(gameState)
            actions=gameState.getLegalActions(agentIndex)
            num_action=float(len(actions))
            value=0
            for action in actions:
                if agentIndex==gameState.getNumAgents()-1:
                    tmp1,tmp2=self.maxAction(depth-1,gameState.generateSuccessor(agentIndex,action))
                    value+=1/num_action *tmp1
                else:
                    value+=1/num_action*self.minAction(depth,gameState.generateSuccessor(agentIndex,action),agentIndex+1)
            return value




def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    nearest_ghost_dis = 1e9
    for ghost_state in newGhostStates:
        ghost_x, ghost_y = ghost_state.getPosition()
        ghost_x = int(ghost_x)
        ghost_y = int(ghost_y)
        if ghost_state.scaredTimer == 0:
            nearest_ghost_dis = min(nearest_ghost_dis,\
                                    manhattanDistance((ghost_x, ghost_y), newPos))
    
    
    food = currentGameState.getFood()
    minFoodDist=1000000
    secondFoodDist=10000000
    if food[newPos[0]][newPos[1]]:
        minFoodDist = 0
    else:
        foodDistances = [
            manhattanDistance(newPos, (x, y))
            for x in range(food.width)
            for y in range(food.height)
            if food[x][y]
        ]
        minFoodDist = min(foodDistances, default=0)
    print(minFoodDist)
    return currentGameState.getScore()+nearest_ghost_dis-minFoodDist
    # return 0

# Abbreviation
better = betterEvaluationFunction
