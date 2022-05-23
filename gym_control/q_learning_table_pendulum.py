
import random
import statistics
import argparse
import sys
from decimal import Decimal
from collections import defaultdict
import gym
from replay_memory import Transition, CircularArrayBuffer
from pympler import classtracker, asizeof
import numpy as np
np.random.seed(10086)


gamma = 0.9
alpha = 0.2
INITIAL_EPSILON = 0.5
DROP_RATE = 0.5
EPISODES_PER_DROP = 500000


class GameInterface(object):
    def __init__(self, env):
        self.env = env
        self.action_dim = 9

    def _quantize(self, state):
        return tuple([Decimal(float(el)).quantize(Decimal(".01")) for el in state])

    def reset(self):
        return self._quantize(self.env.reset())

    def step(self, action):
        
        state, reward, done, info = self.env.step(action)
        
        return self._quantize(state), reward, done, info


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

    @property
    def size_in_mb(self):
        return asizeof.asizeof(self.storage) / 2 ** 20

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

def async_Q(episode, env, q_values, max_tryout=100):
    state = env.reset()
    done = False
    total_reward = 0
    steps=0
    while not done and total_reward < 200:
        action = get_next_action(episode, env, q_values, state, True)
        next_state, reward, done, info = env.step(action)
        q_values.update(state, action, next_state, reward)
        total_reward += reward
        state = next_state
        if steps>max_tryout:
            break
    return total_reward

def EBU(episode, env, q_values, reversed, memory, max_tryout=100):
    """Sample-Efficient Deep Reinforcement Learning via Episodic Backward Update. NIPS 2019"""
    
    done = False
    total_reward = 0
    steps=0
    states, actions, rewards, next_states= [], [], [], []
    state = env.reset()
    while not done and total_reward < 200:
        action = get_next_action(episode, env, q_values, state, is_training=True)
        next_state, reward, done, info = env.step(action)
        
        if steps>max_tryout:
            break
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        
        state = next_state
    memory.insert_traj([states, actions, rewards, next_states]) 
    states, actions, rewards, next_states = memory.retrieve_traj()
    if reversed:
        states=states[::-1]
        actions=actions[::-1] 
        rewards=rewards[::-1]
        next_states=next_states[::-1]
    for state, action, reward, next_state in zip(states, actions, rewards, next_states):
        q_values.update(state, action, next_state, reward)
    return total_reward

def EpiQ_rex(episode, env, q_values, reversed, max_tryout=100):
    """ICLR 2022"""
    state = env.reset()
    done = False
    total_reward = 0
    steps=0
    states, actions, rewards, next_states= [], [], [], []
    while not done and total_reward < 200:
        action = get_next_action(episode, env, q_values, state, is_training=True)
        next_state, reward, done, info = env.step(action)
        
        if steps > max_tryout:
            break
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        state = next_state

    if reversed:
        states=states[::-1]
        actions=actions[::-1] 
        rewards=rewards[::-1]
        next_states=next_states[::-1]
    for state, action, reward, next_state in zip(states, actions, rewards, next_states):
        q_values.update(state, action, next_state, reward)
    return total_reward


def treeRER(episode, env, q_values, L, reversed, memory, max_tryout=100):
    state = env.reset()
    done = False
    total_reward = 0
    steps=0
    
    while not done and total_reward < 200:
        action = get_next_action(episode, env, q_values, state, is_training=True)
        next_state, reward, done, info = env.step(action)
        memory.insert_transition(Transition(state=state, reward=reward, action=action,
                                            next_state=next_state, done=done, epi=0, step=steps))
        total_reward += reward
        state = next_state
        if steps>max_tryout:
            break
    
    batches = np.max([1, int(steps / L)])
    for bi in range(batches):
        sub_traj = memory.retrieve(batch_size=1, L=L, reversed=reversed)
        for trans in sub_traj:
            state, next_state,action, reward = trans[0].state, trans[0].next_state, trans[0].action, trans[0].reward
            q_values.update(state, action, next_state, reward)
    return total_reward


def run_episode(episode, env, q_values, is_training, max_tryout=100):
    state = env.reset()
    done = False
    total_reward = 0
    steps=0
    while not done and total_reward < 200:
        action = get_next_action(episode, env, q_values, state, is_training)
        next_state, reward, done, info = env.step(action)
        if is_training:
            q_values.update(state, action, next_state, reward)
        total_reward += reward
        state = next_state
        if steps>max_tryout:
            break
    return total_reward


def get_next_action(episode, env, q_values, state, is_training):
    if random.random() < get_effective_epsilon(episode) and is_training:
        action = np.random.choice([n for n in range(9)])
        real_action = (action - 4) / 2  # -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0
        return real_action
    else:
        maxed_idx = 0
        for i in range(env.action_space.n):
            if  q_values.get(state, i) >  q_values.get(state, maxed_idx):
                maxed_idx = i
        real_action = (maxed_idx - 4) / 4
        return real_action

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


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    MAX_EPISODES = args.episodes
    env = GameInterface(gym.make(args.env_name))
    qTable = QTable()
    is_training = True
    memory = CircularArrayBuffer(capacity=100000)
    tr = classtracker.ClassTracker()
    tr.track_object(memory)
    for episode in range(MAX_EPISODES):
        if args.algo =="async":
            async_Q(episode, env, qTable, args.max_tryout)
        if args.algo =="EpiQ-rex":
            EpiQ_rex(episode, env, qTable, reversed=args.reversed, max_tryout=args.max_tryout)
        elif args.algo =="treeRER":
            treeRER(episode, env, qTable, args.L, args.reversed, memory, args.max_tryout)
        elif args.algo == "EBU":
            EBU(episode, env, qTable, reversed=args.reversed, memory=memory, max_tryout=args.max_tryout)
        if episode > args.freeze and episode % args.evaluation_step==0:
            max_reward, median_reward, mean_reward, min_reward = evaluate_policy(env, qTable,args.play_times, args.max_tryout)
            print(f"{episode}  {max_reward}  {mean_reward}")
