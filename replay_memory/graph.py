import numpy.random as random
from collections import deque

from replay_memory.utils import Transition


class Node(object):
    def __init__(self, state):
        self.state = state

    def __repr__(self):
        return "(s={})".format(self.state)


class GraphBuffer(object):
    "Topological Experience Replay. ICLR 2022"

    def __init__(self, capacity):
        self.name = "Directed Graph for Trajectories"
        self.capacity = capacity
        self.memory = dict()

    def insert_transition(self, one_trans):
        """Saves a transition."""

        if one_trans.state not in self.memory:
            self.memory[one_trans.state] = dict()

        self.memory[one_trans.state][one_trans.next_state] = (
            one_trans.action, one_trans.reward)

    def bfs(self, L):
        cur_st = random.choice(list(self.memory.keys()))
        traj = []
        while len(traj) < L:
            if cur_st not in self.memory:
                break
            next_st = random.choice(list(self.memory[cur_st].keys()))
            action, reward = self.memory[cur_st][next_st]
            traj.append(Transition(cur_st, action, reward,
                        next_st, None, None, None))

            cur_st = next_st
        return traj

    def random_walk(self, L):
        cur_st = random.choice(list(self.memory.keys()))
        traj = []
        while len(traj) < L:
            next_st = random.choice(list(self.memory[cur_st].keys()))
            action, reward = self.memory[cur_st][next_st]
            traj.append(Transition(cur_st, action, reward,
                        next_st, None, None, None))

            cur_st = next_st
        return traj

    def retrieve(self, batch_size, L, reversed, use_random_walk):
        rand_experiences = [[None, ]*batch_size for _ in range(L)]
        for bi in range(batch_size):
            if use_random_walk:
                retrieved_traj = self.random_walk(L)
            else:
                retrieved_traj = self.bfs(L)
            for l, one_trans in enumerate(retrieved_traj):
                rand_experiences[l][bi] = one_trans
        if reversed:
            return rand_experiences[::-1]
        else:
            return rand_experiences

    def __len__(self):
        sumed = 0
        for x in self.memory:
            sumed += len(self.memory[x])
        return sumed
