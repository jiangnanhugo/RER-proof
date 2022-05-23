import gym
import numpy as np
from numpy import random
import argparse
import math
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from replay_memory import Transition, CircularArrayBuffer

device = 'cpu' if torch.cuda.is_available() else 'cpu'

criterion = nn.SmoothL1Loss()

# hyper-parameter
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 50
steps_done = 0


# Q(S,a)
class DQN(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=50, action_dim=2):
        super(DQN, self).__init__()
        self.DNN= torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, action_dim))

    def forward(self, x):
        return self.DNN(x)


def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() <= eps_threshold:
        return random.randint(0, 1)
    else:
        with torch.no_grad():
            s = Variable(torch.Tensor(state.reshape(1, 4)).to(device=device, dtype=torch.float))
            out = policy_net(s).max(1)[1].data.cpu().numpy().tolist()[0]
        return out


def optimize_model(memory, policy_net, optimizer, batch_size, L, reversed=False, use_double_dqn=False):
    if len(memory) < 5:
        return
    episodics = memory.retrieve(batch_size=batch_size, L=L, reversed=reversed)

    for trans in episodics:
        q_update(trans, policy_net, optimizer, batch_size=batch_size, use_double_dqn=use_double_dqn)



def q_update(batch, policy_net, optimizer, batch_size, use_double_dqn):
    states, actions, rewards, next_states, dones = Transition.unzip(batch)

    state_batch = Variable(torch.Tensor(states).to(device=device, dtype=torch.float))
    action_batch = Variable(torch.LongTensor(actions).to(device=device, dtype=torch.long))
    reward_batch = Variable(torch.Tensor(rewards).to(device=device, dtype=torch.float))
    next_states_batch = Variable(torch.Tensor(next_states).to(device=device, dtype=torch.float))
    state_action_values = policy_net(state_batch).gather(1, action_batch.view(-1, 1))
    if use_double_dqn:
        _, next_state_actions = policy_net(state_batch).max(1, keepdim=True)
        next_state_values = target_net(next_states_batch).gather(1, next_state_actions).squeeze()
        for i in range(batch_size):
            if dones[i]:
                next_state_values.data[i] = 0
    else:
        next_state_values = target_net(next_states_batch).max(1)[0].detach()
        for i in range(batch_size):
            if dones[i]:
                next_state_values.data[i] = 0

    expected_state_action_values = reward_batch + (GAMMA * next_state_values)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def play_average(play_time, max_tryout):
    total_reward = []
    for i in range(play_time):
        test_state = env.reset()
        reward=0
        for j in range(max_tryout):
            inputs = Variable(torch.Tensor(test_state.reshape(1, 4)).to(device=device, dtype=torch.float))
            test_action = policy_net(inputs).max(1)[1].data.cpu().numpy().tolist()[0]
            test_state, test_reward, test_done, _ = env.step(test_action)
            reward += test_reward
                    
            if test_done:
                break
            total_reward.append(reward)
    return total_reward
    

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='CartPole-v1', help='Game name')
    parser.add_argument('--use_double_q', action="store_true", default=False)
    parser.add_argument('--reversed', action="store_true", default=False)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--episodes', type=int, default=250000)
    parser.add_argument('--algo', type=str, default='rer', help="Q learning algorithm")
    
    parser.add_argument('--freeze', type=int, default=100)
    parser.add_argument('--evaluation_step', type=int, default=20)
    parser.add_argument('--max_tryout', type=int, default=2 ** 12)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--play_times', type=int, default=10)
    parser.add_argument('--L', type=int, default=2, help="length of reverse tree_experience_replay")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    episode_slots = [x for x in range(0, args.episodes, args.evaluation_step)]
    env = gym.make(args.env_name)
    policy_net = DQN(hidden_dim=args.hidden_dim)
    target_net = DQN(hidden_dim=args.hidden_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    for episode in range(args.episodes):
        observation = env.reset()
        state = observation
        done = False
        steps = 0
        memory = CircularArrayBuffer(capacity=10000)
        while not done:
            action = select_action(state)  # choose action
            observation, reward, done, info = env.step(action)
            next_state = observation
            # Store the transition in memory
            memory.insert_transition(
                Transition(state=state, action=action, reward=reward, next_state=next_state,
                           done=done, epi=episode, step=steps))
            steps += 1
            # Move to the next state
            state = next_state
            if steps > args.max_tryout:
                break
        optimize_model(memory, policy_net, optimizer, args.batch_size, args.L, args.reversed, args.use_double_q)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode > args.freeze and episode in episode_slots:
                rewards = play_average(args.play_times, args.max_tryout)
                print("{} {} {:.4f}".format(episode, np.mean(rewards), np.std(rewards)))


