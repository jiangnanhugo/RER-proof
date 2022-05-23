import itertools
import pandas as pd
import numpy as np



class TowersOfHanoi(object):
    def __init__(self, state):
        """"State" is a tuple of length N, where N is the number of discs,
                and the elements are peg indices in [0,1,2]"""
        self.state = state  #
        self.discs = len(self.state)

    def discs_on_peg(self, peg):
        return [disc for disc in range(self.discs) if self.state[disc] == peg]

    def move_allowed(self, move):
        discs_from = self.discs_on_peg(move[0])
        discs_to = self.discs_on_peg(move[1])
        if discs_from:
            return (min(discs_to) > min(discs_from)) if discs_to else True
        else:
            return False

    def get_moved_state(self, move):
        if self.move_allowed(move):
            disc_to_move = min(self.discs_on_peg(move[0]))
        moved_state = list(self.state)
        moved_state[disc_to_move] = move[1]
        return tuple(moved_state)


# Generates the reward matrix for the Towers of Hanoi game as a Pandas DataFrame
def generate_reward_matrix(num_disks):
    # N is the number of discs
    states = list(itertools.product(list(range(3)), repeat=num_disks))
    moves = list(itertools.permutations(list(range(3)), 2))
    R = pd.DataFrame(index=states, columns=states, data=-np.inf)
    for state in states:
        tower = TowersOfHanoi(state=state)
        for move in moves:
            if tower.move_allowed(move):
                next_state = tower.get_moved_state(move)
                R[state][next_state] = 0
    final_state = tuple([2] * num_disks)  # Define final state as all discs being on the last peg
    R[final_state] += 100  # Add a reward for all moves leading to the final state
    terminal_state = len(R.keys())
    return R.values, terminal_state