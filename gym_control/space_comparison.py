
import random
import statistics
import argparse
import sys
from decimal import Decimal
from collections import defaultdict
import gym
from replay_memory import Transition, CircularArrayBuffer, TrieBuffer, GraphBuffer
from pympler import classtracker, asizeof
import numpy as np
np.random.seed(10086)


gamma = 0.9
alpha = 0.2
INITIAL_EPSILON = 0.5
DROP_RATE = 0.5
EPISODES_PER_DROP = 500000

def size_in_mb(storage):
    return asizeof.asizeof(storage) / 2 ** 20

class GameInterface(object):
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space

    def _quantize(self, state):
        return tuple([Decimal(float(el)).quantize(Decimal(".01")) for el in state])

    def reset(self):
        return self._quantize(self.env.reset())
    
    def reset2(self):
        return self._quantize(self.env.reset()), self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self._quantize(state), reward, done, info
    
    def step2(self, action):
        state, reward, done, info = self.env.step(action)
        return self._quantize(state), state, reward, done, info


class QTable:
    def __init__(self):
        self.storage = defaultdict(int)
        self.reset_stats()

    @property
    def hit_rate(self):
        if self.access_count == 0:
            return 0
        return self.hit_count / self.access_count

    @property
    def nonzero_hit_rate(self):
        if self.access_count == 0:
            return 0
        return self.nonzero_hit_count / self.access_count

    

    def reset_stats(self):
        self.access_count = 0
        self.hit_count = 0
        self.nonzero_hit_count = 0

    def get(self, state, action):
        self.access_count += 1
        if (state, action) in self.storage:
            self.hit_count += 1
        q_value = self.storage[(state, action)]
        if q_value != 0:
            self.nonzero_hit_count += 1
        return q_value

    def update(self, state, action, next_state, reward):
        curr_q = self.get(state, action)
        discounted_next_q = gamma * max([self.get(next_state, a) for a in [0, 1]])
        q_error = alpha * (reward + discounted_next_q - curr_q)
        updated_curr_q = curr_q + q_error
        self.storage[(state, action)] = updated_curr_q




def get_next_action(episode, q_values, state, is_training):
    if random.random() < get_effective_epsilon(episode) and is_training:
        return random.choice([0, 1])
    else:
        return 0 if q_values.get(state, 0) > q_values.get(state, 1) else 1


def evaluate_policy(env, q_values, play_times, max_tryout):

    is_training = False
    q_values.reset_stats()
    rewards = [run_episode(i, env, q_values, is_training, max_tryout) for i in range(play_times)]
    max_reward = max(rewards)
    mean_reward = statistics.mean(rewards)
    median_reward = statistics.median(rewards)
    min_reward = min(rewards)
    return max_reward, median_reward, mean_reward, min_reward
    
    


def get_effective_epsilon(episode):
    return INITIAL_EPSILON * (DROP_RATE ** (episode / EPISODES_PER_DROP))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='Pendulum-v1', help='Game name')
    parser.add_argument('--reversed', action="store_true", default=False)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--episodes', type=int, default=10000000)
    parser.add_argument('--algo', type=str, default='rer', help="Q learning algorithm")
    
    parser.add_argument('--freeze', type=int, default=100)
    parser.add_argument('--evaluation_step', type=int, default=1000)
    parser.add_argument('--max_tryout', type=int, default=2 ** 12)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--play_times', type=int, default=10)
    parser.add_argument('--L', type=int, default=2, help="length of reverse tree_experience_replay")
    args = parser.parse_args()
    return args

def async_Q(MAX_EPISODES, env, q_values, max_tryout=100):
    
    array=[]
    # tra = classtracker.ClassTracker()
    # tra.track_object(array)
    memory = CircularArrayBuffer(capacity=2**32)
    trie = TrieBuffer()
    graph_buffer = GraphBuffer(capacity=2**32)
    tr = classtracker.ClassTracker()
    tr.track_object(memory)
        
    trb = classtracker.ClassTracker()
    trb.track_object(trie)
        
    trg = classtracker.ClassTracker()
    trg.track_object(graph_buffer)
    for episode in range(MAX_EPISODES):
        steps=0
        state, state_raw = env.reset2()
        done = False
        total_reward = 0
        episodic_trajectory = [state_raw.tobytes(),]
        while not done and total_reward < 200:
            action = get_next_action(episode, q_values, state, True)
            next_state, next_state_raw, reward, done, info = env.step2(action)
            q_values.update(state, action, next_state, reward)
            total_reward += reward
            # print(type(state_raw), type(next_state), type(next_state_raw), type(reward), type(done))
            # print(type(action))
            one_trans = Transition(state=state_raw.tobytes(), reward=int(reward), action=action, 
                                    next_state=next_state_raw.tobytes(), done =None)
            
            memory.insert_transition_unlimited(one_trans)
            graph_buffer.insert_transition(one_trans)
            episodic_trajectory.append(int(action))
            episodic_trajectory.append(int(reward))
            episodic_trajectory.append(next_state_raw.tobytes())

            
            state = next_state
            state_raw = next_state_raw
            if steps>max_tryout:
                break
            steps+=1
        trie.insert_trajectory(episodic_trajectory)
        if episode % 1000 == 0:
            tr.create_snapshot()
            trb.create_snapshot()
            trg.create_snapshot()
            tr.stats.print_summary()
            trb.stats.print_summary()
            trg.stats.print_summary()
            print(len(memory.memory))
            # print("{:.4f}".format(size_in_mb(array)))
            print("-"*80)
    return total_reward

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    MAX_EPISODES = args.episodes
    env = GameInterface(gym.make(args.env_name))
    qTable = QTable()
    is_training = True
    
    
    async_Q(MAX_EPISODES, env, qTable, args.max_tryout)
        
