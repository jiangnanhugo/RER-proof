''' Reinforcement learning of the Towers of Hanoi game.
Reference: Watkins and Dayan, "Q-Learning", Machine Learning, 8, 279-292 (1992).'''

from numpy import random
import numpy as np
import argparse
from tqdm import tqdm

from environments.TowerofHonoiGame import generate_reward_matrix

import pandas as pd
from replay_memory import Transition, CircularArrayBuffer, TrieBuffer, GraphBuffer
from pympler import classtracker, asizeof

np.random.seed(10086)


def size_in_mb(storage):
    return asizeof.asizeof(storage) / 2 ** 20


class tabularQ(object):
    def __init__(self, Rewards, terminal):
        self.Q = np.zeros(Rewards.shape)
        self.R = Rewards
        self.terminal = terminal

    def learn_async_Q(self, gamma, alpha, episodes, max_tryout):
        states = list(range(self.R.shape[0]))
        state = np.random.choice(states)  # Randomly select initial state
        for epidx in range(episodes[-1]):
            done = False
            steps = 0
            while not done:
                # Generate a list of possible next states
                next_states = np.where(self.R[state, :] >= 0)[0]
                # Randomly select next state from the list of possible next states
                next_state = np.random.choice(next_states)
                if next_state == self.terminal or steps > max_tryout:
                    done = True
                # Update Q-values
                target = self.R[state, next_state] + \
                    gamma * np.max(self.Q[next_state, :])
                self.Q[state, next_state] = (1 - alpha) * self.Q[state, next_state] \
                    + alpha * target

                if np.max(self.Q) > 0:  # Normalize Q to its maximum value
                    self.Q /= np.max(self.Q)
                steps += 1

            if epidx in episodes:
                policy = self.get_policy()
                means, stds = self.play_average(
                    policy, args.play_times, args.max_tryout)
                print("{}: {} \xB1 {}".format(epidx, means, stds))

    def learn_epiQ_rex(self, gamma, alpha, episodes, freeze, L, reversed, max_tryout=300):
        states = list(range(self.R.shape[0]))

        array_buffer = CircularArrayBuffer(capacity=2 ** 32)
        trie_buffer = TrieBuffer()
        graph_buffer = GraphBuffer(capacity=2 ** 32)

        for epidx in range(episodes[-1]):
            state = np.random.choice(states)  # Randomly select initial state
            steps = 0
            done = False
            episodic_trajectory = [state, ]
            while not done:
                # Generate a list of possible next states
                next_states = np.where(self.R[state, :] >= 0)[0]
                # Randomly select next state from the list of possible next states
                next_state = np.random.choice(next_states)
                if next_state == self.terminal or steps >= max_tryout:
                    done = True
                one_tran = Transition(state=state, reward=self.R[state, next_state], action=None,
                                      next_state=next_state, done=done, epi=None, step=None)
                array_buffer.insert_transition(one_tran)
                graph_buffer.insert_transition(one_tran)
                episodic_trajectory.append(next_state)
                episodic_trajectory.append(self.R[state, next_state])
                state = next_state
                steps += 1
            # print("done exploration")
            trie_buffer.insert_trajectory(episodic_trajectory)
            batches = np.max([1, int(steps / L)])
            for bi in tqdm(range(batches), leave=False):
                sub_traj = array_buffer.retrieve(
                    batch_size=1, L=L, reversed=reversed)
                # print(sub_traj)
                for trans in sub_traj:
                    st, nxt_st, reward = trans[0].state, trans[0].next_state, trans[0].reward
                    self.Q[st, nxt_st] = (1 - alpha) * self.Q[st, nxt_st] + \
                        alpha * (reward + gamma * np.max(self.Q[nxt_st, :]))
                if np.max(self.Q) > 0:
                    # Normalize Q to its maximum value
                    self.Q /= np.max(self.Q)
            # print("done policy update")

            # evaluation
            if epidx > freeze and epidx in episodes:
                idx = episodes.index(epidx)
                policy = self.get_policy()
                means, stds = self.play_average(
                    policy, args.play_times, args.max_tryout)
                print("reward: {} {} {:.4f}".format(epidx, means, stds))

            print("space: {:.2f}/{} {:.2f}/{} {:.2f}".format(
                size_in_mb(array_buffer), len(array_buffer),
                  size_in_mb(graph_buffer), len(graph_buffer),
                  size_in_mb(trie_buffer)))

    def get_policy(self):
        Q_allowed = pd.DataFrame(self.Q)[pd.DataFrame(self.R) >= 0].values
        policy = []
        for i in range(Q_allowed.shape[0]):
            row = Q_allowed[i, :]
            sorted_vals = np.sort(row)
            sorted_vals = sorted_vals[~np.isnan(sorted_vals)][::-1]
            sorted_args = row.argsort()[np.where(~np.isnan(sorted_vals))][::-1]
            max_args = [sorted_args[i] for i, val in enumerate(
                sorted_vals) if val == sorted_vals[0]]
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
    parser.add_argument('--gamma', type=float, default=0.9,
                        help="discounting factor")
    parser.add_argument('--alpha', type=float,
                        default=0.5, help="learning rate")
    parser.add_argument('--algo', type=str, default='rer',
                        help="Q learning algorithm")
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--freeze', type=int, default=100)
    parser.add_argument('--evaluation_step', type=int, default=20)
    parser.add_argument('--max_tryout', type=int, default=2 ** 12)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--play_times', type=int, default=2)
    parser.add_argument('--num_pillars', type=int, default=3)
    parser.add_argument('--num_disks', type=int, default=9)
    parser.add_argument('--reversed', action="store_true", default=False)
    parser.add_argument('--L', type=int, default=2,
                        help="length of reverse tree_experience_replay")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    print(args)
    R, terminal = generate_reward_matrix(args.num_disks)

    episodes = [x for x in range(0, args.episodes, args.evaluation_step)]
    for i in range(args.repeats):
        Qmodel = tabularQ(R, terminal)
        if args.algo == 'rer':
            Qmodel.learn_epiQ_rex(gamma=args.gamma, alpha=args.alpha,
                                  L=args.L, episodes=episodes, freeze=args.freeze,
                                  max_tryout=args.max_tryout, reversed=args.reversed)
        elif args.algo == 'async':
            Qmodel.learn_async_Q(gamma=args.gamma, alpha=args.alpha, episodes=episodes,
                                 max_tryout=args.max_tryout)
