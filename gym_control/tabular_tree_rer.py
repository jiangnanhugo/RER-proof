''' Reinforcement learning of the Towers of Hanoi game.
Reference: Watkins and Dayan, "Q-Learning", Machine Learning, 8, 279-292 (1992).'''
from numpy import random
import numpy as np
import argparse
from collections import defaultdict

np.random.seed(10086)
import gym
import pandas as pd
from replay_memory import Transition, CircularArrayBuffer

# Observation: 
#     Type: Box(4)
#     Num	Observation            Min         Max
#     0	Cart Position             -4.8            4.8 -> terminate when |cart position| > 2.4
#     1	Cart Velocity             -Inf            Inf
#     2	Pole Angle    -24 deg = -0.42 radian  24 deg = -0.42 radian -> terminate when |pole angle| > 12deg
#     3	Pole Velocity At Tip      -Inf            Inf
# or reach timestep 500

class tabularQ(object):
    def __init__(self, env):
        self.env = env
        # self.alpha = 0.1
        # self.gamma = 1
        self.epsilon = 0.5
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.0


        self.cart_pos_bin = np.linspace(-2.4, 2.4, num=1000)[1:-1]
        self.cart_vel_bin = np.linspace(-3, 3, num=1000)[1:-1]
        self.pole_ang_bin = np.linspace(-0.21, 0.21, num=1000)[1:-1]
        self.pole_vel_bin = np.linspace(-2.0, 2.0, num=1000)[1:-1]
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def state_coding(self, state):
        cart_pos = np.digitize([state[0]], self.cart_pos_bin)[0]
        cart_vel = np.digitize([state[1]], self.cart_vel_bin)[0]
        pole_ang = np.digitize([state[2]], self.pole_ang_bin)[0]
        pole_vel = np.digitize([state[3]], self.pole_vel_bin)[0]
        return (cart_pos, cart_vel, pole_ang, pole_vel)

    def act(self, state):
        coded_state = self.state_coding(state)
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[coded_state])
        return action        
    
    
    def learn_async_Q(self, gamma, alpha, episodes, max_tryout):
         
        for epidx in range(episodes[-1]):
            done = False
            steps = 0
            state = env.reset() 
            while not done:
                action = agent.act(state)
                state_next, reward, done, info = env.step(action) # take a random action

                coded_state = self.state_coding(state)
                coded_state_next = self.state_coding(state_next)
                if steps > max_tryout:
                    done = True
                # Update Q-values
                target = reward + gamma * max(self.Q[coded_state_next])
                self.Q[coded_state][action] += alpha * (target - self.Q[coded_state][action])
                state = state_next
                steps += 1
                if done:
                    env.close()

            if epidx in episodes:
                idx = episodes.index(epidx)
                rewards = self.play_average(args.play_times, args.max_tryout)
                print("{} {} {:.4f}".format(epidx, np.mean(rewards), np.std(rewards)))

    
    def learn_epiQ_rex(self, gamma, alpha, episodes, freeze, L, reversed, max_tryout=300):
        state = env.reset()  # Randomly select initial state
        memory = CircularArrayBuffer(capacity=100000)
        for epidx in range(episodes[-1]):
            steps = 0
            done = False
            while not done:
                action = agent.act(state)
                state_next, reward, done, info = env.step(action) # take a random action

                coded_state = self.state_coding(state)
                coded_state_next = self.state_coding(state_next)
                if steps > max_tryout:
                    done = True
                memory.insert_transition(Transition(state=coded_state, reward=reward, action=action,
                                                    next_state=coded_state_next, done=done, epi=epidx, step=steps))
                state = state_next
                steps += 1
            batches = np.max([1, int(steps / L)])
            for bi in range(batches):
                sub_traj = memory.retrieve(batch_size=1, L=L, reversed=reversed)
                # print(sub_traj)
                for trans in sub_traj:
                    st, nxt_st, action, reward = trans[0].state, trans[0].next_state, trans[0].action, trans[0].reward
                    target = reward + gamma * max(self.Q[coded_state_next])
                    self.Q[coded_state][action] += alpha * (target - self.Q[coded_state][action])
                

            # evaluation
            if epidx > freeze and epidx in episodes:
                idx = episodes.index(epidx)
                moves, rewards = self.play_average(args.play_times, args.max_tryout)
                print("{} {} {:.4f}".format(epidx, np.mean(rewards), np.std(rewards)))
        


    def play_average(self, play_times, max_tryout):
        moves = np.zeros(play_times)
        rewards = np.zeros(play_times)
        for i in range(play_times):
            moves[i] = 0
            state = self.env.reset()
            done = False
            while not done:
                coded_state = self.state_coding(state)
                action = np.argmax(self.Q[coded_state])
                state_next, reward, done, info = env.step(action) 
                if moves[i] > max_tryout:
                    done = True
                state = state_next
                moves[i] += 1
                rewards[i] += reward
                if done:
                    env.close()
        return rewards

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='CartPole-v1')
    parser.add_argument('--gamma', type=float, default=0.9, help="discounting factor")
    parser.add_argument('--alpha', type=float, default=0.1, help="learning rate")
    parser.add_argument('--algo', type=str, default='rer', help="Q learning algorithm")
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--freeze', type=int, default=100)
    parser.add_argument('--evaluation_step', type=int, default=20)
    parser.add_argument('--max_tryout', type=int, default=2 ** 12)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--play_times', type=int, default=10)
    parser.add_argument('--reversed', action="store_true", default=False)
    parser.add_argument('--L', type=int, default=2, help="length of reverse tree_experience_replay")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    print(args)
    env = gym.make(args.env_name)
    print("done with game settings....")
    episodes = [x for x in range(0, args.episodes, args.evaluation_step)]
    ####
    
    for i in range(args.repeats):
        agent = tabularQ(env)
        if args.algo == 'rer':
            agent.learn_epiQ_rex(gamma=args.gamma, alpha=args.alpha,
                                  L=args.L, episodes=episodes, freeze=args.freeze,
                                  max_tryout=args.max_tryout, reversed=args.reversed)
        elif args.algo == 'async':
            agent.learn_async_Q(gamma=args.gamma, alpha=args.alpha, episodes=episodes,
                                 max_tryout=args.max_tryout)
    