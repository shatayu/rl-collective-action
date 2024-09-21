# We set β = 0.4 and A = 0.9. (A) X = 0.3, (B) X = 0.4, and (B) X = 0.5

iteNum = 10000
N = 3 # number of agents EXCLUDING the RL agent

a = 1.6 # multiply factor for PGG

std = 0.2 # std of distribution from which contribution is drawn
beta = 0.4
A = 1.0
X = 0.4 # cooperativeness criteria for Bush-Mosteller algorithm
COOPERATIVE_CONSTANT_FOR_REWARD = 0.4 # cooperativeness criteria for reward function

import numpy as np
import gymnasium as gym
from ray.rllib.utils import try_import_tf
import math

tf = try_import_tf()

class RLWithBushMostellerEnv(gym.Env):
    def __init__(self, config):
        self.num_rounds_hidden = config['num_rounds_hidden']
        self.reward_function = config['reward_function']
        self.tmax = config['tmax'] # number of rounds

        assert self.reward_function in ['sum', 'proportion'], 'Invalid reward function'
    
        self.aveCont = [0.0] * self.tmax
        self.net = self.completeNet()
        self.pt = [0.0] * N
        self.At = [0.0] * N
        self.st = [0.0] * N
        self.at = [0.0] * (N + 1)
        self.payoff = [0.0] * N
        self.current_round = 0
        self.all_at = [([0.0] * (N + 1)) for _ in range(self.tmax)]

        self.action_space = gym.spaces.Discrete(101)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.tmax, N + 1), dtype=np.float32)

        self.initialize(self.net, self.payoff, self.at, self.pt, self.st, self.At)

    def initialize(self, net, payoff, at, pt, st, At):
        for i in range(N):
            payoff[i] = 0
            pt[i] = np.random.rand()
            at[i] = np.random.normal() * std + pt[i] # initial contribution

            while at[i] < 0 or at[i] > 1:
                at[i] = np.random.normal() * std + pt[i] # discard irrational contribution
            
            At[i] = A
            st[i] = 0

        self.all_at[self.current_round] = at.copy()

    def reset(self, *, seed=None, options=None):
        self.aveCont = [0.0] * self.tmax
        self.net = self.completeNet()
        self.pt = [0.0] * N
        self.At = [0.0] * N
        self.st = [0.0] * N
        self.at = [0.0] * (N + 1)
        self.payoff = [0.0] * N
        self.current_round = 0
        self.all_at = [([0.0] * (N + 1)) for _ in range(self.tmax)]

        self.initialize(self.net, self.payoff, self.at, self.pt, self.st, self.At)

        return self.get_state(), {}
        
    def completeNet(self):
        net = [[] for i in range(N + 1)]

        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    net[i].append(j)
        
        return net

    def updatePGG(self, net, payoff, at, pt, st, At):
        # compute payoffs
        for i in range(len(payoff)):
            pool = 0
            for j in range(len(net[i])): # collect contributions from neighbors
                pool += at[net[i][j]] * a
            
            pool += at[i] * a
            payoff[i] = 1 - at[i] + pool / (len(net[i]) + 1)
        
        # update pt[]
        for i in range(N):
            st[i] = math.tanh(beta * (payoff[i] - At[i]))

            if (at[i] >= X): # cooperation
                if st[i] >= 0:
                    pt[i] = pt[i] + (1 - pt[i]) * st[i]
                else:
                    pt[i] = pt[i] + pt[i] * st[i]
            else: # defected
                if st[i] >= 0:
                    pt[i] = pt[i] - pt[i] * st[i]
                else:
                    pt[i] = pt[i] - (1 - pt[i]) * st[i]
        
        # draw contribution for next round
        for i in range(N):
            at[i] = np.random.normal() * std + pt[i]
            while at[i] < 0 or at[i] > 1:
                at[i] = np.random.normal() * std + pt[i]

        self.all_at[self.current_round] = at.copy()
    
    def step(self, action_input):
        action = action_input / 100.0
        if self.current_round == 0:
            self.at[N] = action
            self.all_at[0][N] = action
        elif self.current_round < self.tmax:
            self.at[N] = action
            self.updatePGG(self.net, self.payoff, self.at, self.pt, self.st, self.At)
            self.aveCont[self.current_round] += np.mean(self.at)

        self.current_round += 1
        
        return self.get_state(), self.get_reward(), self.current_round >= self.tmax, False, {}
    
    def get_state(self):
        pass

    def get_reward(self):
        if self.current_round < self.tmax:
            return 0
        else:
            return (
                sum(sum(at[:N]) for at in self.all_at) # sum reward
                if self.reward_function == 'sum'
                else sum([len(list(filter(lambda x: x > 0.5, at[:3]))) for at in self.all_at]) # proportional reward
            )

class RLWithBushMostellerWholeGameEnv(RLWithBushMostellerEnv):
    def __init__(self, *args):
        super().__init__(*args)
        self.observation_space = gym.spaces.Box(low=-1, high=self.tmax + 1, shape=(self.tmax * (N + 1) + 1, ), dtype=np.float32)

    def get_state(self):
        game = np.concatenate([
            np.array(self.all_at).reshape(-1, 1),
            np.array([self.current_round]).reshape(1, 1)
        ])

        game[:((N + 1) * self.num_rounds_hidden)] = -1

        return game.reshape(1, -1)[0].astype(np.float32)

def run_n_rl_games(algo, reward_function, n):
    rewards = []
    final_states = []

    for i in range(n):
        total_reward, state = run_one_rl_game(algo, reward_function)
        rewards.append(total_reward)
        final_states.append(state)   

    return rewards, final_states 

def run_one_rl_game(algo, reward_function):
    total_reward = 0.0
    env = RLWithBushMostellerWholeGameEnv({
        'num_rounds_hidden': 0,
        'reward_function': reward_function
    })
    state, _ = env.reset()

    done = False
    while not done:
        action = algo.compute_single_action(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    
    return total_reward, state