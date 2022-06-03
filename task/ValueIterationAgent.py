from agent import Agent
import numpy as np

# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        self.V = {s : 0 for s in states}
        self.pi = {s : None for s in states}
        # ************

        for i in range(iterations):
            newV = {}
            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                number_actions = len(actions)
                # **************
                # TODO 2.1. b)
                if len(actions)<1:
                    newV[s] = 0
                else:
                    Qval_action_dict = {a : 0 for a in actions}
                    for a in actions:
                        nstate_prob = self.mdp.getTransitionStatesAndProbs(s,a)
                        for tup in nstate_prob:
                            next_state, prob = tup
                            Qval_action_dict[a] += prob * (self.mdp.getReward(s,a,next_state) + self.discount * self.V[next_state])
                    newV[s] = max(Qval_action_dict.values())
                    self.pi[s] = max(Qval_action_dict, key=Qval_action_dict.get)
                # Update value function with new estimate
            self.V = newV
        

                # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return self.V[state]
        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        nstate_prob = self.mdp.getTransitionStatesAndProbs(state,action)
        Q_val = 0
        for tup in nstate_prob:
            next_state, prob = tup
            Q_val += prob * (self.mdp.getReward(state,action,next_state)+self.discount*self.V[next_state])
        return Q_val
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        # **********
        # TODO 2.4
        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None
        else:
            return self.pi[state]
        # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
