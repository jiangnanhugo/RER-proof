

class Transition(object):
    def __init__(self, state, action, reward, next_state, done=None, epi=None, step=None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.epi = epi
        self.step = step

    def __repr__(self):
        return "(e{},t{})".format(self.epi, self.step)

    @staticmethod
    def unzip(batched_transition):
        state, action, reward, next_state, done = [], [], [], [], []
        for bi in batched_transition:
            state.append(bi.state)
            action.append(bi.action)
            reward.append(bi.reward)
            next_state.append(bi.next_state)
            done.append(bi.done)
        return state, action, reward, next_state, done
