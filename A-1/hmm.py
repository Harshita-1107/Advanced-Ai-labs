from enum import IntEnum 

class SuspectState(IntEnum):
    Planning = 1,
    Scouting = 2,
    Burglary = 3,
    Migrating = 4,
    Misc = 5

class Daytime(IntEnum):
    Day = 6
    Evening = 7
    Night = 8

class Action(IntEnum):
    Roaming = 9,
    Eating = 10,
    Home = 11,
    Untracked = 12,

class Observation:
    def __init__(self, d:Daytime, a:Action) -> None:
        self.daytime = d
        self.action = a


# This function reads the string file 
# that contains the sequence of observations
# that is required to learn the HMM model.
def ReadDataset() -> list:
    # Converts integer to Daytime enum
    def getDay(p: int) -> Daytime:
        if p == 6:
            return Daytime.Day
        elif p == 7:
            return Daytime.Evening
        elif p == 8:
            return Daytime.Night
        else:
            assert False, 'Unexpected Daytime!'

    # Converts integer to Action enum
    def getAct(p: int) -> Action:
        if p == 9:
            return Action.Roaming
        elif p == 10:
            return Action.Eating
        elif p == 11:
            return Action.Home
        elif p == 12:
            return Action.Untracked
        else:
            assert False, 'Unexpected Action!'
    filepath = 'database.txt' 
    with open(filepath, 'r') as file:
        seq_count = int(file.readline())
        seq_list = []
        for _ in range(seq_count):
            w = file.readline().split(' ')
            len = int(w[0])
            seq_i = []
            for k in range(0, len):
                idx = (k*2) + 1
                day = int(w[idx])
                act = int(w[idx + 1])
                o = Observation(getDay(day), getAct(act))
                seq_i.append(o)
            seq_list.append(seq_i)
    return seq_list


#  --------------Do not change anything above this line---------------

class HMM:
    def __init__(self, data: list[Observation], states: list[SuspectState]) -> None:
        self.states = states
        self.data = data

        self.state_count = len(states)
        self.observation_count = len(Action)

        self.transition_matrix = [[0 for i in range(self.state_count)] 
                                  for j in range(self.state_count)]
        self.emission_matrix = [[0 for i in range(self.observation_count)] 
                                 for j in range(self.state_count)]
        self.initial_probabilities = [0 for i in range(self.state_count)]

    def A(self, a: SuspectState, b: SuspectState) -> float:
        return self.transition_matrix[a.value - 1][b.value - 1]

    def B(self, a: SuspectState, b: Action) -> float:
        return self.emission_matrix[a.value - 1][b.daytime - 6][b.action - 9]

    def Pi(self, a: SuspectState) -> float:
        return self.initial_probabilities[a.value - 1]
   
        
    def Train(self, iterations: int) -> None:
        for i in range(iterations):
            alpha = self.ForwardProcedure()
            beta = self.BackwardProcedure()

            xi = self.ComputeXi(alpha, beta)
            gamma = self.ComputeGamma(alpha, beta)

            self.UpdateTransitionMatrix(xi, gamma)
            self.UpdateEmissionMatrix(gamma, self.data)
            self.UpdateInitialProbabilities(gamma)
     
    def ForwardProcedure(self) -> list[list[float]]:
        alpha = [[0 for i in range(self.observation_count)] for j in range(self.state_count)]

        for i in range(self.state_count):
            alpha[i][0] = self.Pi(self.states[i]) * self.B(self.states[i], self.data[0])

        for t in range(1, self.observation_count):
            for i in range(self.state_count):
                prob = 0
                for j in range(self.state_count):
                    prob += alpha[j][t - 1] * self.A(self.states[j], self.states[i])
                    alpha[i][t] = prob * self.B(self.states[i], self.data[t])

        return alpha

    def BackwardProcedure(self) -> list[list[float]]:
        beta = [[0 for i in range(self.observation_count)] for j in range(self.state_count)]

        for i in range(self.state_count):
            beta[i][self.observation_count - 1] = 1

        for t in range(self.observation_count - 2, -1, -1):
            for i in range(self.state_count):
                prob = 0
                for j in range(self.state_count):
                    prob += self.A(self.states[i], self.states[j]) * self.B(self.states[j], self.data[t + 1]) * beta[j][t + 1]
                beta[i][t] = prob

        return beta

    def ComputeXi(self, alpha: list[list[float]], beta: list[list[float]]) -> list[list[list[float]]]:
        xi = [[[0 for i in range(self.observation_count - 1)] for j in range(self.state_count)] for k in range(self.state_count)]

        for t in range(self.observation_count - 1):
            denom = 0
            for i in range(self.state_count):
                for j in range(self.state_count):
                    denom += alpha[i][t] * self.A(self.states[i], self.states[j]) * self.B(self.states[j], self.data[t + 1]) * beta[j][t + 1]

        for i in range(self.state_count):
            for j in range(self.state_count):
                numer = alpha[i][t] * self.A(self.states[i], self.states[j]) * self.B(self.states[j], self.data[t + 1]) * beta[j][t + 1]
                xi[i][j][t] = numer / denom

        return xi


    def ComputeGamma(self, alpha: list[list[float]], beta: list[list[float]]) -> list[list[float]]:
        gamma = [[0 for i in range(self.observation_count)] for j in range(self.state_count)]

        for t in range(self.observation_count):
            denom = 0
            for i in range(self.state_count):
                denom += alpha[i][t] * beta[i][t]

            for i in range(self.state_count):
                gamma[i][t] = (alpha[i][t] * beta[i][t]) / denom

        return gamma

    def UpdateTransitionMatrix(self, xi: list[list[list[float]]], gamma: list[list[float]]) -> None:
        for i in range(self.state_count):
            for j in range(self.state_count):
                numer = 0
                denom = 0
                for t in range(self.observation_count - 1):
                    numer += xi[t][i][j]
                    denom += gamma[i][t]
                self.transition_matrix[i][j] = numer / denom

    def UpdateEmissionMatrix(self, gamma: list[list[float]], data: list[Observation]) -> None:
        for i in range(self.state_count):
            for j in range(self.observation_count):
                numer = 0
                denom = 0
                for t in range(self.observation_count):
                    if data[t].action == self.data[j].action:
                        numer += gamma[i][t]
                    denom += gamma[i][t]
                self.emission_matrix[i][j] = numer / denom

    def UpdateInitialProbabilities(self, gamma: list[list[float]]) -> None:
        for i in range(self.state_count):
            self.initial_prob[i] = gamma[0][i] / sum(gamma[0][i] for i in range(self.num_states))

# Part I
# ---------

def LearnModel(dataset: list) -> HMM:
    data = ReadDataset()
    states = list(SuspectState)

    model = HMM(data, states)
    model.Train(50)

    print('Transition matrix:')
    for i in range(len(states)):
        for j in range(len(states)):
            print(f'{model.A(states[i], states[j]):.2f}', end=' ')
        print()

    print('\nEmission matrix:')
    for i in range(len(states)):
        for j in range(len(data)):
            print(f'{model.B(states[i], data[j]):.2f}', end=' ')
        print()

    print('\nInitial probabilities:')
    for i in range(len(states)):
        print(f'{model.Pi(states[i]):.2f}', end=' ')
    print()

# Part II
# ---------

import numpy as np

def Liklihood(model: HMM, obs_list: list) -> float:
    T = len(obs_list)
    alpha = np.zeros((T, model.N))
    
    alpha[0, :] = model.prior * model.emission[:, obs_list[0].idx]
    for t in range(1, T):
        for j in range(model.N):
            alpha[t, j] = alpha[t-1].dot(model.transition[:, j]) * model.emission[j, obs_list[t].idx]
    
    return np.sum(alpha[T-1, :])

# // Part III
# //---------

def GetHiddenStates(model: HMM, obs_list: list) -> list:
    T = len(obs_list)
    delta = np.zeros((T, model.N))
    psi = np.zeros((T, model.N), dtype=int)
    
    delta[0, :] = model.prior * model.emission[:, obs_list[0].idx]
    for t in range(1, T):
        for j in range(model.N):
            delta[t, j] = np.max(delta[t-1] * model.transition[:, j]) * model.emission[j, obs_list[t].idx]
            psi[t, j] = np.argmax(delta[t-1] * model.transition[:, j])
    
    states = [np.argmax(delta[T-1, :])]
    for t in range(T-2, -1, -1):
        states.append(psi[t+1, states[-1]])
    
    return states[::-1]
