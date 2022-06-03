import numpy as np
from agent import Agent


# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # Policy initialization
        # ******************
        # TODO 1.1.a)
        self.V =  {s : 0 for s in states}
        print("State value functions of the states are:",self.V)
        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}
        print("Policy for the each state is:",self.pi)
        counter = 0
        counter_2 = 0
        while True:
            # Policy evaluation
            for i in range(iterations):
                newV = {}
                
                for s in states:
                    a = self.pi[s]
                    # *****************
                    # TODO 1.1.b)
                    if self.mdp.isTerminal(s):
                        newV[s] = 0.0
                    else:
                        trans_prob = self.mdp.getTransitionStatesAndProbs(s,a)
                        currV = 0
                        for j in trans_prob:
                            next_state, prob = j
                            reward = self.mdp.getReward(s,a,next_state)
                            currV += prob * (reward + self.discount*self.V[next_state])
                        newV[s] = currV
                # update value estimate
                self.V = newV
                counter_2 =+ 1
                if (self.V[(4, 0)] == 0):
                    print("ZERO")
                else : 
                    print("NON-ZERO, ROUND : ",counter_2)

                # ******************

            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    Qval_action_dict = {a : 0 for a in actions}
                    for a in actions:
                        nstate_prob = self.mdp.getTransitionStatesAndProbs(s,a)
                        v = 0
                        for tup in nstate_prob:
                            next_state, prob = tup
                            v += prob * (self.mdp.getReward(s,a,next_state) + self.discount * self.V[next_state])
                        Qval_action_dict[a] = v
                    updated_policy_action = max(Qval_action_dict, key=Qval_action_dict.get)
                    self.pi[s] = updated_policy_action
            
                    if old_action != updated_policy_action:
                       policy_stable = False

                    # ****************
            counter += 1

            
            if policy_stable:
                break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        #TODO 1.2.
        return self.V[state] 
        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # TODO 1.3.
        n_state_prob_2 = self.mdp.getTransitionStatesAndProbs(state, action)
        Q_val = 0 
        for tup in n_state_prob_2:
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
        # TODO 1.4.
        return self.pi[state]
        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
