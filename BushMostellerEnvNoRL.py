from RLWithBushMostellerEnv import tmax, std, iteNum, X, beta, a, N, A
import numpy as np
import math

class BushMostellerEnvNoRL():
    def __init__(self):
        self.N = N + 1
        self.aveCont = [0.0] * tmax
        self.net = self.completeNet()
        self.pt = [0.0] * self.N
        self.At = [0.0] * self.N
        self.st = [0.0] * self.N
        self.at = [0.0] * self.N
        self.payoff = [0.0] * self.N

    def initialize(self, net, payoff, at, pt, st, At):
        for i in range(self.N):
            payoff[i] = 0
            pt[i] = np.random.rand()
            at[i] = np.random.normal() * std + pt[i] # initial contribution

            while at[i] < 0 or at[i] > 1:
                at[i] = np.random.normal() * std + pt[i] # discard irrational contribution
            
            At[i] = A
            st[i] = 0

    def completeNet(self):
        net = [[] for i in range(self.N)]

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    net[i].append(j)
        
        return net

    def updatePGG(self, net, payoff, at, pt, st, At):
        # compute payoffs

        for i in range(self.N):
            pool = 0
            for j in range(len(net[i])): # collect contributions from neighbors
                pool += at[net[i][j]] * a
            
            pool += at[i] * a
            payoff[i] = 1 - at[i] + pool / (len(net[i]) + 1)
        
        # update pt[]
        for i in range(self.N):
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
        for i in range(self.N):
            at[i] = np.random.normal() * std + pt[i]
            while at[i] < 0 or at[i] > 1:
                at[i] = np.random.normal() * std + pt[i]

    def main(self):
        for ite in range(iteNum):
            self.initialize(self.net, self.payoff, self.at, self.pt, self.st, self.At)
            for t in range(tmax):
                self.updatePGG(self.net, self.payoff, self.at, self.pt, self.st, self.At)
                self.aveCont[t] += np.mean(self.at)

    def run_one_game(self):
        all_at = []
        self.initialize(self.net, self.payoff, self.at, self.pt, self.st, self.At)
        for t in range(tmax):
            self.updatePGG(self.net, self.payoff, self.at, self.pt, self.st, self.At)
            all_at.append(self.at.copy())
        return all_at

def run_n_baseline_games(n):
    env = BushMostellerEnvNoRL()
    return [env.run_one_game() for _ in range(n)]
    