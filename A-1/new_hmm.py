from hmm import *

# Part IV
# ---------

class UpdatedSuspectState(IntEnum):
    Exploring = -1,
    Planning = 0,
    Escaping = 1,
    Scouting = 2,
    Burglary = 3,
    Migrating = 4,
    Misc = 5
    

class Updated_HMM:
    
    def __init__(self, data: list[Observation], states: list[UpdatedSuspectState]) -> None:
        self.states = states
        self.data = data

        self.state_count = len(states)
        self.observation_count = len(Action)

        self.transition_matrix = [[0 for i in range(self.state_count)] 
                                  for j in range(self.state_count)]
        self.emission_matrix = [[0 for i in range(self.observation_count)] 
                                 for j in range(self.state_count)]
        self.initial_probabilities = [0 for i in range(self.state_count)]

    def A(self, a: UpdatedSuspectState, b: UpdatedSuspectState) -> float:
        return self.transition_matrix[a.value - 1][b.value - 1]

    def B(self, a: UpdatedSuspectState, b: Observation) -> float:
        return self.emission_matrix[a.value - 1][b.action.value - 9]

    def Pi(self, a: UpdatedSuspectState) -> float:
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



def LearnUpdatedModel(dataset: list) -> Updated_HMM:
    data = ReadDataset()
    states = list(UpdatedSuspectState)

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

def LiklihoodUpdated(model: Updated_HMM, obs_list: list) -> float:
    T = len(obs_list)
    alpha = np.zeros((T, model.N))
    
    alpha[0, :] = model.prior * model.emission[:, obs_list[0].idx]
    for t in range(1, T):
        for j in range(model.N):
            alpha[t, j] = alpha[t-1].dot(model.transition[:, j]) * model.emission[j, obs_list[t].idx]
    
    return np.sum(alpha[T-1, :])

def GetUpdatedHiddenStates(model: Updated_HMM, obs_list: list) -> list:
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

if __name__ == "__main__":
    database = ReadDataset()
    old_model = LearnModel(database)
    new_model = LearnUpdatedModel(database)
    
    obs_list = []
    w = [6, 11, 7, 10, 8, 9, 6, 11, 7, 12, 8, 12, 6, 12, 7, 9, 8, 11]
    for k in range(0, 9):
        idx = (k * 2)
        day = int(w[idx])
        act = int(w[idx + 1])
        o = Observation(get_Day(day), get_Act(act))
        obs_list.append(o)
    p = Liklihood(old_model, obs_list)
    q = LiklihoodUpdated(new_model, obs_list)

    print(p)
    print(q)
    old_states = GetHiddenStates(old_model, obs_list)
    new_states = GetUpdatedHiddenStates(new_model, obs_list)
    print(old_states)
    print(new_states)
