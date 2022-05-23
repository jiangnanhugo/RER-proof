import numpy.random as random


class CircularArrayBuffer(object):
    """ array data structure used in DQN 2015 nature."""

    def __init__(self, capacity):
        self.name = "Circular Buffer Array"
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert_transition(self, one_trans):
        """Saves one transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = one_trans
        self.position = (self.position + 1) % self.capacity

    def insert_traj(self, traj):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = traj
        self.position = (self.position + 1) % self.capacity

    def insert_transition_unlimited(self, one_trans):
        self.memory.append(one_trans)

    def retrieve_traj(self):
        rnd_idx = random.randint(len(self.memory))
        return self.memory[rnd_idx]

    def retrieve(self, batch_size, L, reversed):
        if L == 1:
            return random.sample(self.memory, size=batch_size)
        else:
            rand_experiences = [[] for _ in range(L)]
            rnd_idxs = random.randint(len(self.memory), size=batch_size)
            for l in range(L):
                rand_experiences[l] = [
                    self.memory[(ri + l) % len(self.memory)] for ri in rnd_idxs]

            if reversed:
                return rand_experiences[::-1]
            else:
                return rand_experiences

    def __len__(self):
        return len(self.memory)
