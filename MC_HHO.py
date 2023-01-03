import random
import time
import copy
import math
import numpy as np


def load_graph(filename):
    lines = open(filename).readlines()
    n_node = int(lines[0].split()[0])
    G = [[0 for _ in range(n_node)] for _ in range(n_node)]

    for i in range(1, len(lines)):
        a = int(lines[i].split()[0]) - 1
        b = int(lines[i].split()[1]) - 1
        v = int(lines[i].split()[2])
        G[a][b] = G[b][a] = v

    return G, n_node


def adjustment(arr, lb, ub):
    arr = np.clip(abs(arr), lb, ub)
    return arr


def mutation(rl, x, hawks):
    F = random.random()
    randomHawkIndexList = random.sample(range(0, hawks - 1), 4)
    return rl + F * (x[randomHawkIndexList[0]] - x[randomHawkIndexList[1]]) + (x[randomHawkIndexList[2]] - x[randomHawkIndexList[3]])


def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step


def objf(G, H):
    G = np.array(flatten(G), dtype='float')
    return sum(G - H) / 2


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def random_subset(v):
    s = [i for i in range(v)]
    out = set()
    for el in s:
        if random.randint(0, 1) == 0:
            out.add(el)
    return out


def randomHawksGeneration(v, h):
    r = sum([math.factorial(v) // (math.factorial(k) * math.factorial(v - k)) for k in range(1, v)])
    randHawks = list()

    while True:
        if len(randHawks) == h:
            break

        x = list(random_subset(v))

        if len(x) not in [0, v]:
            if len(randHawks) >= r:
                randHawks.append(x)
            elif x not in randHawks:
                randHawks.append(x)

    return randHawks


def HHO(G, lb, ub, dim, SearchAgents_no, Max_iter):
    Rabbit_Location = np.zeros(dim * dim)
    Rabbit_Energy = float("-inf")

    X = []
    initialSubsets = randomHawksGeneration(dim, SearchAgents_no)

    for i in range(SearchAgents_no):
        print(i)
        A = initialSubsets[i]
        B = [k for k in V if k not in A]
        G_new = copy.deepcopy(G)

        for x in A:
            for y in B:
                G_new[x][y] = G_new[y][x] = 0

        G_new = flatten(G_new)
        X.append(G_new)
        # print(G_new)
    X = np.array(X, dtype='float')
    startTime = time.time()
    t = 0
    while t < Max_iter:
        E1 = 2 * (1 - (t / Max_iter))

        for i in range(0, SearchAgents_no):
            fitness = objf(G, X[i, :])

            if fitness > Rabbit_Energy:
                print(fitness, t)
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()

        for i in range(0, SearchAgents_no):
            temp = X[i].copy()
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0

            if abs(Escaping_Energy) >= 1:
                q = random.random()
                rand_Hawk_index = i
                X_rand = X[rand_Hawk_index, :]

                # To not selecting the same Hawk
                while rand_Hawk_index == i:
                    rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                    X_rand = X[rand_Hawk_index, :]

                if q < 0.5:
                    X[i, :] = abs(X_rand - random.random() * abs(X_rand - 2 * random.random() * X[i, :]))

                elif q >= 0.5:  # Negative Values
                    X[i, :] = abs(Rabbit_Location - random.random() * ((ub - lb) * random.random() + lb))

            elif abs(Escaping_Energy) < 1:
                r = random.random()

                if abs(Escaping_Energy) < 0.5 <= r:
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs(
                        Rabbit_Location - X[i, :])  # Negative when X[i] > Rabbit_Location

                if r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())  # range [0, 2]
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :])

                if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                    X1 = adjustment(X1, lb, ub)

                    if objf(G, X1) > fitness:
                        X[i, :] = X1.copy()
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * abs(
                            Jump_strength * Rabbit_Location - X[i, :])
                        if objf(G, X2) > fitness:
                            X[i, :] = X2.copy()

                if r < 0.5 and abs(Escaping_Energy) < 0.5:  # Hard besiege Eq. (11) in paper
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(0))
                    X1 = adjustment(X1, lb, ub)

                    if objf(G, X1) > fitness:  # improved move?
                        X[i, :] = X1.copy()

                    else:  # Perform levy-based short rapid dives around the rabbit
                        X2 = Rabbit_Location - Escaping_Energy * abs(
                            Jump_strength * Rabbit_Location - X.mean(0))
                        X2 = adjustment(X2, lb, ub)

                        if objf(G, X2) > fitness:
                            X[i, :] = X2.copy()

            X[i] = mutation(Rabbit_Location, X, SearchAgents_no)
            X[i, :] = adjustment(X[i], lb, ub)

        print(['At iteration ' + str(t) + ' the best fitness is ' + str(Rabbit_Energy)])
        t += 1
    endTime = time.time()
    print('Average Time elapsed : ', (endTime - startTime) / t)
    print('-----------')


if __name__ == '__main__':
    G, dim = load_graph('sample_graph.txt')
    V = [i for i in range(dim)]
    Max_iter = 100
    Search_agent_no = 100
    lower, upper = 0, 1

    HHO(G, lower, upper, dim, Search_agent_no, Max_iter)
