import argparse
import numpy as np
from collections import namedtuple
from correlated_q import soccer
from tqdm import tqdm

np.random.seed(666) # let him who hath understanding reckon the number of the seed

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('niter', metavar='N', type=int, nargs=1)

    args = vars(parser.parse_args())

    niter = args['niter'][0]

    # make a struct for representing states
    State = namedtuple('State', 'ball_pos alpha beta')

    # build state tuples (in one line, thanks python!)
    S = [State(p,a,b) for p in [0,1] for a in range(8) for b in range(8) if a != b]

    # generate a mapping from labeled actions to movement vectors in index space
    a_vectors = {'N':-4,'S':+4,'E':+1,'W':-1,'STICK':0}

    # define the action set
    a_labels = ['N', 'S', 'E', 'W', 'STICK']
    a_basis = [a_vectors[a] for a in a_labels]
    A = [(a,b) for a in a_basis for b in a_basis]

    # initialize the transition model
    P = np.zeros((len(S), len(A), len(S)))

    # initialize the reward model
    R = np.zeros((len(S), 2))

    # define the goal states
    G_a = [0,4] # goal sub-states for A
    G_b = [3,7] # goal sub-states for B

    # {idx(s) : s in S, s.ball_pos = a and s.alpha in G_a}
    a_win = [i for i, s in enumerate(S) if s.ball_pos == 0 and s.alpha in G_a]
    b_win = [i for i, s in enumerate(S) if s.ball_pos == 1 and s.beta in G_b]
    a_fail = [i for i, s in enumerate(S) if s.ball_pos == 0 and s.alpha in G_b]
    b_fail = [i for i, s in enumerate(S) if s.ball_pos == 1 and s.beta in G_a]

    # define the reward model
    R[a_win, :] = [100,-100]
    R[b_win, :] = [-100,100]
    R[a_fail, :] = [-100,100]
    R[b_fail, :] = [100,-100]

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            # get the set of non-zero probability s' given s,a
            s_p_dist = soccer.s_prime_distribution(S[i], list(A[j]))

            for s_prime, prob in s_p_dist:
                k = [idx for idx, s in enumerate(S)
                     if s.ball_pos == s_prime.ball_pos
                     and s.alpha == s_prime.alpha
                     and s.beta == s_prime.beta][0]

                P[i,j,k] = prob

    # helper transition lambda returns an s' as a stochastic function of (s,a)
    # in deterministic case, this is always the same for a given (s,a)
    transition = lambda s,a : np.random.choice(np.argwhere(P[s,a,:]).flatten())

    # epsilon greedy action selector
    e_greedy = lambda Q,e : np.random.choice(Q.shape[1]) if np.random.rand() >= e else Q[s,:].argmax()


    timeout = 50 # episode length timeout threshold, shoud never occur

    # joint action matrix
    A_joint = np.arange(25).reshape(5,5)

    # initialize Q to zero over the states and joint action space
    Q1 = np.ones((len(S), len(a_basis),len(a_basis)))
    Q2 = np.ones((len(S), len(a_basis),len(a_basis)))

    # initialize V functions:
    V1 = np.ones(len(S))
    V2 = np.ones(len(S))

    # initialize pi functions
    pi_1 = np.ones((len(S), len(a_basis))) / len(a_basis)
    pi_2 = np.ones((len(S), len(a_basis))) / len(a_basis)

    # define temporal discount factor
    gamma = .9

    # learning rate
    alpha = lambda T : 1.0 / (0.05 * float(T) + 1.0)

    # epsilon-greedy term, choose randomly with probability episolon
    epsilon = lambda T : 1.0 / (0.01 * float(T) + 1.0)

    # initialize s0
    s0 = [i for i,s in enumerate(S) if s.ball_pos == 1 and s.beta == 1 and s.alpha == 2][0]

    # delta in Q(s,a)
    ERR = []

    for T in tqdm(range(niter)):
        s = s0 # always initalize an episode in s0
        q_sa = Q1[s0,1,4] #player A moves south, player B sticks

        for t in range(timeout):
            # epsilon-greedily select an action
            a1 = soccer.epi_greedy(pi_1, s, epsilon(T))
            a2 = soccer.epi_greedy(pi_2, s, epsilon(T))
            # action is defined jointly by the two agents
            a = A_joint[a1,a2]

            # query transition model to obtain s'
            s_prime = transition(s, a)

            # query the reward model to obtain r
            r1 = R[s_prime, 0]
            r2 = R[s_prime, 1]

            # update Q
            Q1[s,a1,a2] = (1 - alpha(T)) * Q1[s,a1,a2] + alpha(T) * (r1 + gamma * V1[s_prime])
            Q2[s,a2,a1] = (1 - alpha(T)) * Q2[s,a2,a1] + alpha(T) * (r2 + gamma * V2[s_prime])

            # use LP to find pi
            pi_1[s,:] = soccer.pi_solve(pi_1, Q1, s)
            pi_2[s,:] = soccer.pi_solve(pi_2, Q2, s)

            # update V
            V1[s] = np.min(Q1[s,:,:].sum(axis=0) * pi_1[s,a1])
            V1[s] = np.min(Q2[s,:,:].sum(axis=0) * pi_2[s,a2])

            # update s
            s = s_prime

            # terminate when a goal is made
            if r1 != 0 or r2 != 0: break

        ERR.append(np.abs(Q1[s0,1,4] - q_sa)) #player A moves south, player B sticks


