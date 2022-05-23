import util, random

"""
Agent.py defines what an agent is (in terms of the Agent interface below)
and specifies your ValueIterationAgent and QLearningAgent, where you will
fill in code.
"""

class Agent:
  """
  This is an *interface* (outline) for what an agent should do. 
  
  PLEASE DO NOT EDIT THE CONTENTS OF THIS CLASS
  
  Instead, skip ahead to ValueIterationAgent after reading this.
  """

  def getAction(self, state):
    """
    For the given state, get the agent's chosen
    action.  The agent knows the legal actions.
    """
    util.raiseNotDefined()

  def getValue(self, state):
    """
    Get the value of the state (denoted V(s) in the lecture slides)
    """
    util.raiseNotDefined()

  def getQValue(self, state, action):
    """
    Get the q-value of the state action pair (denoted Q(s,a) in lecture slides)
    """
    util.raiseNotDefined()

  def getPolicy(self, state):
    """
    Get the policy recommendation for the state (denoted pi(s) in lecture)
    
    The current policy for a state is the optimal action under the current 
    set of Q-values.  Note that because some agents take random exploration
    moves (e.g., q-learning in Q6), the policy may not return the same 
    action as "getAction".
    """
    util.raiseNotDefined()

  def update(self, state, action, nextState, reward):
    """
    Update the internal state of a learning agent
    according to the (state, action, nextState)
    transistion and the given reward.
    """
    util.raiseNotDefined()


class RandomAgent(Agent):
  """
  Clueless random agent, used only for testing.
  """

  def __init__(self, actionFunction):
    self.actionFunction = actionFunction

  def getAction(self, state):
    return random.choice(self.actionFunction(state))

  def getValue(self, state):
    return 0.0

  def getQValue(self, state, action):
    return 0.0

  def getPolicy(self, state):
    "NOTE: 'random' is a special policy value; don't use it in your code."
    return 'random'

  def update(self, state, action, nextState, reward):
    pass


class ValueIterationAgent(Agent):

  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
    Your value iteration agent should take an mdp on
    construction, run the indicated number of iterations
    and then act according to the resulting policy.
    
    Some useful mdp methods you will use:
        mdp.getStates()
        mdp.getPossibleActions(state)
        mdp.getTransitionStatesAndProbs(state, action)
        mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getValue(self, state):
    """
    Return the value of the state (after the indicated
    number of value iteration passes).
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getQValue(self, state, action):
    """
    Look up the q-value of the state action pair
    (after the indicated number of value iteration
    passes).  Note that value iteration does not
    necessarily create this quantity and you may have
    to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getPolicy(self, state):
    """
    Look up the policy's recommendation for the state
    (after the indicated number of value iteration passes).
    
    This method should return exactly one legal action for each state.
    You may break ties any way you see fit. The getPolicy method is used 
    for display purposes & in the getAction method below.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getAction(self, state):
    """
    Return the action recommended by the policy.  We have provided this
    for you.
    """
    return self.getPolicy(state)

  def update(self, state, action, nextState, reward):
    """
    Not used for value iteration agents!
    """
    pass


class QLearningAgent(Agent):

  def __init__(self, actionFunction, discount = 0.9, learningRate = 0.1, epsilon = 0.2):
    """
    A Q-Learning agent knows nothing about the mdp on
    construction other than a function mapping states to actions.
    The other parameters govern its exploration
    strategy and learning rate.
    
    The actionFunction takes a state and returns a set of legal actions.
    It is the mdp.getPossibleActions method for the underlying mdp.
    """
    self.setLearningRate(learningRate)
    self.setEpsilon(epsilon)
    self.setDiscount(discount)
    self.actionFunction = actionFunction
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getValue(self, state):
    """
    Look up the current value of the state.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getQValue(self, state, action):
    """
    Look up the current q-value of the state action pair.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getPolicy(self, state):
    """
    Look up the current recommendation for the state.  Note that this
    is the optimal policy under the current q-values, and should not
    reference epsilon or include random moves.  You may break ties any
    way you see fit.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def getAction(self, state):
    """
    Choose an action: this will require that your agent balance
    exploration and exploitation as appropriate.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

  def update(self, state, action, nextState, reward):
    """
    Update parameters in response to the observed transition.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

 # THESE NEXT METHODS ARE NEEDED TO WIRE YOUR AGENT UP TO THE CRAWLER GUI
  def setLearningRate(self, learningRate):
        self.learningRate = learningRate

  def setEpsilon(self, epsilon):
    self.epsilon = epsilon

  def setDiscount(self, discount):
    self.discount = discount
    
########################
########################
##                    ##
## ANALYSIS QUESTIONS ##
##                    ##
########################
########################

def question2():
  return None

def question3():
  return None
  
def question4a():
  return None

def question4b():
  return None

def question4c():
  return None

def question4d():
  return None

def question4e():
  return None

def question7():
  return None

if __name__ == '__main__':
  print 'Answers to analysis questions:'
  import agent
  for q in ['2','3','4a','4b','4c','4d','4e','7']:
    response = getattr(agent, 'question' + q)()
    print '  Question %s:\t%s' % (q, str(response))