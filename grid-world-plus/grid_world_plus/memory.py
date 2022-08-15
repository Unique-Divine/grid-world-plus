import numpy as np

"""
idea:
could create a graph that gets the optimal discounted_reward given observations
"""


class Memory:
    def __init__(self):
        self.past_states_reps = []
        self.past_discounted_rewards = []

    def memorize(self, state_representation, discounted_reward):
        self.past_states_reps.append(state_representation)
        self.past_discounted_rewards.append(discounted_reward)
        # TODO memorize next state to model counterfactual exploration

    def val_of_similar(self, state):
        similarity_scores = []
        for ps in self.past_states_reps:
            similarity_scores.append(state @ ps)

        argmax_index = np.argmax(similarity_scores)  # note: may return more than 1 index
        value = self.past_discounted_rewards[argmax_index]
        if len(argmax_index) > 1:
            value = np.mean(value)

        # TODO
        """
        score as confidence measure
        model desire to explore with action in memory 
        e.g. action taken many times should not be taken again
        """
        return value


