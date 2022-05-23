''' Reinforcement learning of the Towers of Hanoi game.
Reference: Watkins and Dayan, "Q-Learning", Machine Learning, 8, 279-292 (1992).'''
from numpy import random
import numpy as np
import argparse

np.random.seed(10086)
from environments.TowerofHonoiGame import *
import pandas as pd
from replay_memory import Transition, CircularArrayBuffer, GraphBuffer
from pympler import classtracker, asizeof
def size_in_mb(storage):
    return asizeof.asizeof(storage) / 2 ** 20

class tabularQ(object):
    def __init__(self, Rewards, terminal):
        self.Q = np.zeros(Rewards.shape)
        self.R = Rewards
        self.terminal = terminal
    
    def learn_async_Q(self, gamma, alpha, episodes, max_tryout):
        means = np.zeros(len(episodes))
        stds = np.zeros(len(episodes))
        states = list(range(self.R.shape[0]))
        state = np.random.choice(states)  # Randomly select initial state
        for epidx in range(episodes[-1]):
            done = False
            steps = 0
            while not done:
                next_states = np.where(self.R[state, :] >= 0)[0]  # Generate a list of possible next states
                # Randomly select next state from the list of possible next states
                next_state = np.random.choice(next_states)
                if next_state == self.terminal or steps > max_tryout:
                    done = True
                # Update Q-values
                self.Q[state, next_state] = (1 - alpha) * self.Q[state, next_state] + \
                                            alpha * (self.R[state, next_state] + gamma * np.max(self.Q[next_state, :]))

                if np.max(self.Q) > 0:
                    self.Q /= np.max(self.Q)  # Normalize Q to its maximum value
                steps += 1

            if epidx in episodes:
                idx = episodes.index(epidx)
                policy = self.get_policy()
                means[idx], stds[idx] = self.play_average(policy, args.play_times, args.max_tryout)
                print("{}: {} \xB1 {}".format(epidx, means[idx], stds[idx]))

    
    def learn_epiQ_rex(self, gamma, alpha, episodes, freeze, L, reversed, max_tryout=300):
        means = np.zeros(len(episodes))
        stds = np.zeros(len(episodes))
        states = list(range(self.R.shape[0]))
        memory = CircularArrayBuffer(capacity=100000)
        for epidx in range(episodes[-1]):
            state = np.random.choice(states)  # Randomly select initial state
            steps = 0
            done = False
            while not done:
                next_states = np.where(self.R[state, :] >= 0)[0]  # Generate a list of possible next states
                # Randomly select next state from the list of possible next states
                next_state = np.random.choice(next_states)
                if next_state == self.terminal or steps >= max_tryout:
                    done = True
                memory.insert_transition(Transition(state=state, reward=self.R[state, next_state], action=None,
                                                    next_state=next_state, done=done, epi=epidx, step=steps))
                state = next_state
                steps += 1
            batches = np.max([1, int(steps / L)])
            for bi in range(batches):
                sub_traj = memory.retrieve(batch_size=1, L=L, reversed=reversed)
                
                for trans in sub_traj:
                    st, nxt_st, reward = trans[0].state, trans[0].next_state, trans[0].reward
                    self.Q[st, nxt_st] = (1 - alpha) * self.Q[st, nxt_st] + \
                                         alpha * (reward + gamma * np.max(self.Q[nxt_st, :]))
                if np.max(self.Q) > 0:
                    self.Q /= np.max(self.Q)  # Normalize Q to its maximum value

            # evaluation
            if epidx > freeze and epidx in episodes:
                idx = episodes.index(epidx)
                policy = self.get_policy()
                means[idx], stds[idx] = self.play_average(policy, args.play_times, args.max_tryout)
                print("{} {} {:.4f}".format(epidx, means[idx], stds[idx]))
        return means, stds

    def learn_EBU(self, gamma, alpha, episodes, freeze, L, reversed, max_tryout=300):
        means = np.zeros(len(episodes))
        stds = np.zeros(len(episodes))
        states = list(range(self.R.shape[0]))
        memory = CircularArrayBuffer(capacity=100000)
        for epidx in range(episodes[-1]):
            state = np.random.choice(states)  # Randomly select initial state
            steps = 0
            done = False
            while not done:
                next_states = np.where(self.R[state, :] >= 0)[0]  # Generate a list of possible next states
                # Randomly select next state from the list of possible next states
                next_state = np.random.choice(next_states)
                if next_state == self.terminal or steps >= max_tryout:
                    done = True
                memory.insert_transition(Transition(state=state, reward=self.R[state, next_state], action=None,
                                                    next_state=next_state, done=done, epi=epidx, step=steps))
                state = next_state
                steps += 1
            batches = np.max([1, int(steps / L)])
            for bi in range(batches):
                sub_traj = memory.retrieve(batch_size=1, L=L, reversed=reversed)
                
                for trans in sub_traj:
                    st, nxt_st, reward = trans[0].state, trans[0].next_state, trans[0].reward
                    self.Q[st, nxt_st] = (1 - alpha) * self.Q[st, nxt_st] + \
                                         alpha * (reward + gamma * np.max(self.Q[nxt_st, :]))
                if np.max(self.Q) > 0:
                    self.Q /= np.max(self.Q)  # Normalize Q to its maximum value

            # evaluation
            if epidx > freeze and epidx in episodes:
                idx = episodes.index(epidx)
                policy = self.get_policy()
                means[idx], stds[idx] = self.play_average(policy, args.play_times, args.max_tryout)
                print("{} {} {:.4f}".format(epidx, means[idx], stds[idx]))
        return means, stds
    
    def learn_graph_RER(self, gamma, alpha, episodes, freeze, L, reversed, max_tryout=300, use_random_walk=False):
        means = np.zeros(len(episodes))
        stds = np.zeros(len(episodes))
        states = list(range(self.R.shape[0]))
        Gmemory = GraphBuffer(capacity=100000)
        for epidx in range(episodes[-1]):
            state = np.random.choice(states)  # Randomly select initial state
            steps = 0
            done = False
            while not done:
                next_states = np.where(self.R[state, :] >= 0)[0]  # Generate a list of possible next states
                # Randomly select next state from the list of possible next states
                next_state = np.random.choice(next_states)
                if next_state == self.terminal or steps >= max_tryout:
                    done = True
                Gmemory.insert_transition(Transition(state=state, reward=self.R[state, next_state], action=None,
                                                    next_state=next_state, done=done, epi=epidx, step=steps))
                state = next_state
                steps += 1

            if epidx <= freeze:
                continue
            
            batches = np.min([20, int(steps / L)])
            
            for bi in range(batches):
                sub_traj = Gmemory.retrieve(batch_size=1, L=L, reversed=reversed, use_random_walk=use_random_walk)
                
                for trans in sub_traj:
                    st, nxt_st, reward = trans[0].state, trans[0].next_state, trans[0].reward
                    self.Q[st, nxt_st] = (1 - alpha) * self.Q[st, nxt_st] + \
                                         alpha * (reward + gamma * np.max(self.Q[nxt_st, :]))
                if np.max(self.Q) > 0:
                    self.Q /= np.max(self.Q)  # Normalize Q to its maximum value

            # evaluation
            # print("evaluate!! ",epidx, episodes, epidx in episodes)
            if epidx in episodes:
                idx = episodes.index(epidx)
                policy = self.get_policy()
                means[idx], stds[idx] = self.play_average(policy, args.play_times, args.max_tryout)
                print("{} {} {:.4f}".format(epidx, means[idx], stds[idx]))
        return means, stds

    def get_policy(self):
        Q_allowed = pd.DataFrame(self.Q)[pd.DataFrame(self.R) >= 0].values
        policy = []
        for i in range(Q_allowed.shape[0]):
            row = Q_allowed[i, :]
            sorted_vals = np.sort(row)
            sorted_vals = sorted_vals[~np.isnan(sorted_vals)][::-1]
            sorted_args = row.argsort()[np.where(~np.isnan(sorted_vals))][::-1]
            max_args = [sorted_args[i] for i, val in enumerate(sorted_vals) if val == sorted_vals[0]]
            policy.append(max_args)
        return policy

    def play_average(self, policy, play_times, max_tryout):
        moves = np.zeros(play_times)
        for i in range(play_times):
            moves[i] = 0
            start_state = 0
            end_state = len(policy) - 1
            state = start_state
            while state != end_state:
                state = np.random.choice(policy[state])
                moves[i] += 1
                if moves[i] > max_tryout:
                    break
        return np.mean(moves), np.std(moves)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='Hanoi-v0')
    parser.add_argument('--gamma', type=float, default=0.9, help="discounting factor")
    parser.add_argument('--alpha', type=float, default=0.5, help="learning rate")
    parser.add_argument('--algo', type=str, default='rer', help="Q learning algorithm")
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--freeze', type=int, default=10)
    parser.add_argument('--evaluation_step', type=int, default=20)
    parser.add_argument('--max_tryout', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--play_times', type=int, default=10)
    parser.add_argument('--num_pillars', type=int, default=3)
    parser.add_argument('--num_disks', type=int, default=9)
    parser.add_argument('--reversed', action="store_true", default=False)
    parser.add_argument('--L', type=int, default=2, help="length of reverse tree_experience_replay")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    print(args)
    R, terminal = generate_reward_matrix(args.num_disks)
    print("done with game settings....")
    episodes = [x for x in range(0, args.episodes, args.evaluation_step)]
    ####
    
    for i in range(args.repeats):
        Qmodel = tabularQ(R, terminal)
        if args.algo == 'rer':
            Qmodel.learn_epiQ_rex(gamma=args.gamma, alpha=args.alpha,
                                  L=args.L, episodes=episodes, freeze=args.freeze,
                                  max_tryout=args.max_tryout, reversed=args.reversed)
        elif args.algo == 'EBU':
            Qmodel.learn_EBU(gamma=args.gamma, alpha=args.alpha,
                                  L=args.L, episodes=episodes, freeze=args.freeze,
                                  max_tryout=args.max_tryout, reversed=args.reversed)
        elif args.algo == 'graph':
            Qmodel.learn_graph_RER(gamma=args.gamma, alpha=args.alpha,
                                  L=args.L, episodes=episodes, freeze=args.freeze,
                                  max_tryout=args.max_tryout, reversed=args.reversed)
        elif args.algo == 'async':
            Qmodel.learn_async_Q(gamma=args.gamma, alpha=args.alpha, episodes=episodes,
                                 max_tryout=args.max_tryout)
    