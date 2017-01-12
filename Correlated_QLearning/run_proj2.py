import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
# from correlated_q import soccer
from tqdm import tqdm
#################################################################################
np.random.seed(666)

#makeastructforrepresentingstates
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


#definethegoalstates
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



def collide(first_player, ball_pos, a):
    second_player = 1 - first_player

    # handle the case where the second player is sticking
    if a[second_player] == 0:
        a = [0,0]

    # for all other acitons, only the second player is blocked
    else:
        a[second_player] = 0

    # change possesion when second player has the ball
    bp_prime = 1 - ball_pos if ball_pos == second_player else ball_pos
    return (bp_prime, a)


def s_prime_distribution(s,a):
    '''s in S is a 3-tuple comprising the current state, a in A is an action vector
    containing the actions selected by A and B'''
    # unpack variables
    ball_pos, alpha, beta = s

    # impose boundary constraint (can't fall off the edge)
    if (alpha + a[0] > 7 or alpha + a[0] < 0):
        a[0] = 0
    if (beta + a[1] > 7 or beta + a[1] < 0):
        a[1] = 0

    if not(alpha + a[0] == beta + a[1]):
        return [(State(ball_pos, alpha + a[0], beta + a[1]), 1)]
    else:
        p = []
        # assume A goes first
        bp_prime, a = collide(0,ball_pos,a)
        alpha_prime = alpha + a[0]
        beta_prime = beta + a[1]
        p.append((State(bp_prime, alpha_prime, beta_prime), .5))

        # assume B goes first
        bp_prime, a = collide(1,ball_pos,a)
        alpha_prime = alpha + a[0]
        beta_prime = beta + a[1]
        p.append((State(bp_prime, alpha_prime, beta_prime), .5))

        return p


for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        # get the set of non-zero probability s' given s,a
        s_p_dist = s_prime_distribution(S[i], list(A[j]))

        for s_prime, prob in s_p_dist:
            k = [idx for idx, s in enumerate(S)
                 if s.ball_pos == s_prime.ball_pos
                 and s.alpha == s_prime.alpha
                 and s.beta == s_prime.beta]
            assert len(k) == 1
            k = k[0]

            P[i,j,k] = prob


# for aIdx in range(25):
#     plt.figure(figsize=(5,5))
#     plt.imshow(P[:, aIdx , :], interpolation='nearest')
#     plt.title('Stochastic Transition Matrix: ' + str( aIdx))
#     plt.xlabel('$s\'$',fontsize='20')
#     plt.ylabel('$s$',fontsize='20')
#     plt.colorbar()
#     print A[aIdx]
#     plt.show()

#######################################################################
# Validate definitions by playing games randomly
# rewards = []
# game_len = []
# wins = []
#
# for i in range(100000):
#     # init the state uniformally over possible states
#     s = np.random.randint(56)
#
#     # run until game terminates or timeout is reached
#     for t in range(50):
#         # choose random action get the set of possible s'
#         a = np.random.randint(25)
#         s_primes = np.argwhere(P[s,a,:]).flatten()
#
#         # choose one (in deterministic case, this is always the same)
#         s = np.random.choice(s_primes)
#
#         # compute the rewards
#         R_a = R[s,0]
#         R_b = R[s,1]
#
#         if (R_a != 0 or R_b != 0):
#             wins.append(0) if R_a > R_b else wins.append(1)
#             rewards.append([R_a, R_b])
#             game_len.append(t+1)
#             break
#
# print "Expectation over rewards (should be close to 0) = {}".format(np.mean(rewards, axis=0))
# print "Win average (should be close to .5): {}".format(np.mean(wins))


def revMaxErr(E0):
    E = E0
    a =  -1
    N = len(E)
    for i in range(len(E)):
        j = N-1-i
        E[j] = max(E[j], a)
        a = E[j]
    return E
#################################################################################
#iimport numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
# from correlated_q import soccer
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
# from correlated_q import soccer
 # ###Friend_Q
# #
# # helper transition lambda returns an s' as a stochastic function of (s,a)
# # in deterministic case, this is always the same for a given (s,a)
transition = lambda s,a : np.random.choice(np.argwhere(P[s,a,:]).flatten())
#
# # epsilon greedy action selector
e_greedy = lambda Q,e : np.random.choice(Q.shape[1]) if np.random.rand() >= e else Q[s,:].argmax()

# n_iter = 1000000 # numer of iterations of Q learning
# # timeout = 2500
# timeout = 200
#
# # initialize Q to zero over the states and joint action space
# Q1 = np.zeros((len(S), len(A)))
# Q2 = np.zeros((len(S), len(A)))
#
# # define temporal discount factor
# gamma = .9
#
# # learning rate
# # alpha = lambda T : 1.0 / (0.001 * float(T) + 1.0)
# alpha = lambda T : 1.0 / (0.01* float(T) + 1.0)
#
# # epsilon-greedy term, choose randomly with probability episolon
# epsilon = lambda T : 1.0 / (0.01 * float(T) + 1.0)
#
# # initialize s0
# s0 = [i for i,s in enumerate(S) if s.ball_pos == 1 and s.beta == 1 and s.alpha == 2][0]
#
# # delta in Q(s,a)
# ERR =[]
# ERR2 = []
#
# # ------------------------------------------
# # Action vector selected for measurement in Greenwald & Hall 2003
# a0 = [i for i,a in enumerate(A) if a[0] == 4 and a[1] == 0]
# assert len(a0) == 1
# a0 = a0[0]
# A=[]
# # for T in tqdm(range(n_iter)):
# for T in range(n_iter):
#     s = s0 # always initalize an episode in s0
#     q_sa = Q1[s0,a0]
#     q_a0 = Q1[:,a0].copy()
#     Q1_last = Q1.copy()
#     print "simulating itr: ", T
#     goalFlag = False
#     for t in range(timeout):
#     # while True:
#         # epsilon-greedily select an action
#         # a1 = e_greedy(Q1, epsilon(T))
#         # a2 = e_greedy(Q2, epsilon(T))
#         a1 = e_greedy(Q1, 0.01)
#         a2 = e_greedy(Q2, 0.01)
#
#         # let the agent with the max Q choose
#         a = a1 if Q1[s,a1] > Q2[s,a2] else a2
#
#         # query transition model to obtain s'
#         s_prime = transition(s, a)
#
#         # query the reward model to obtain r
#         r1 = R[s_prime, 0]
#         r2 = R[s_prime, 1]
#
#         # update Q
#         # Q1[s,a] = Q1[s,a] + alpha(T) * (r1 + gamma * (Q1[s_prime,:].max() - Q1[s,a]))
#         # Q2[s,a] = Q2[s,a] + alpha(T) * (r2 + gamma * (Q2[s_prime,:].max() - Q2[s,a]))
#         qqq = Q1[s,a1]
#         Q1[s,a1] = (1 - alpha(T)) * Q1[s,a1] + alpha(T) * ( (1. - gamma) * r1 + gamma * Q1[s_prime,:].max() )
#         Q2[s,a2] = (1 - alpha(T)) * Q2[s,a2] + alpha(T) * ( (1. - gamma) * r1 + gamma * Q2[s_prime,:].max() )
#         ### ZGC
#         ### ERR.append(np.abs(Q1[s,a1] - qqq))
#
#
#         # update s
#         s = s_prime
#         ### terminate when a goal is made
#         if r1 != 0 or r2 != 0:
#             goalFlag = True
#             break
#         # if r1==100 or r2==100:
#         #     break
#         #
#         # curr_err = np.mean(np.abs(Q1 - Q1_last))
#         # # print curr_err
#         # ERR2.append(curr_err)
#     ERR.append(np.abs(Q1[s0,a0] - q_sa))
#     A.append(alpha(T))
#
#
#
#     if goalFlag:
#         curr_err = np.mean(np.abs(Q1[:,a0] - Q1_last[:,a0]))
#         # print curr_err
#         ERR2.append(curr_err)
#
#     # err_single = np.mean(np.abs(Q1[:,a0] - q_a0))
#     # # err_single = (np.abs(Q1[s0,a0] - q_sa))
#     # # print err_single
#     # if err_single > -0.0000000000001:
#     #     print err_single
#     #     ERR.append(err_single)
#
#
# #print ERR
#
# # plt.figure()
# # plt.plot(ERR, 'black')
# # plt.ylim((0, .5))
# # plt.title('Friend-Q')
# # plt.xlabel('Simulation Iteration')
# # plt.ylabel('Q-Value Difference')
# # # plt.xticks(np.arange(0, len(ERR), 250000))
# # # plt.xticks(np.arange(0, len(ERR), 100))
# # plt.show()
#
#
# # ERR = list(np.array(ERR[0:20000]) + 1) + ERR
# ERR3 = revMaxErr(ERR)
#
# plt.figure()
# plt.plot(A)
# # plt.ylim((0, .5))
# plt.title('alpha value for friend-Q')
# plt.xlabel('Simulation Iteration')
# # plt.ylabel('Q-Value Difference')
# # plt.xticks(np.arange(0, len(ERR), 250000))
# # plt.xticks(np.arange(0, len(ERR), 100000))
# plt.show()
#
# plt.figure()
# plt.plot(ERR3, 'red')
# plt.ylim((0, .5))
# plt.title('friend-Q')
# plt.xlabel('Simulation Iteration')
# plt.ylabel('Q-Value Difference')
# # plt.xticks(np.arange(0, len(ERR), 250000))
# plt.xticks(np.arange(0, len(ERR), 100000))
# plt.show()
# quit()

#
#
# # # ################################################################################
# # ###Q-learning
# n_iter = 1000000 # numer of iterations of Q learning
# timeout = 50 # episode length timeout threshold, shoud never occur
#
# # joint action matrix
# A_joint = np.arange(25).reshape(5,5)
#
# # initialize Q to zero over the states and joint action space
# Q1 = np.zeros((len(S), len(a_basis)))
# Q2 = np.zeros((len(S), len(a_basis)))
#
# # define temporal discount factor
# gamma = .9
#
# # learning rate
# alpha = lambda T : 1.0 / (0.00015 * pow(float(T)/9, 1.2) + 1.0)
#
# # epsilon-greedy term, choose randomly with probability episolon
# epsilon = lambda T : 1.0 / (0.01 * float(T) + 1.0)
#
# # initialize s0
# s0 = [i for i,s in enumerate(S) if s.ball_pos == 1 and s.beta == 1 and s.alpha == 2][0]
#
# # delta in Q(s,a)
# ERR = []
#
# # Action vector selected for measurement in Greenwald & Hall 2003
# a0 = [i for i,a in enumerate(A) if a[0] == 4 and a[1] == 0][0]
# A=[]
# for T in tqdm(range(n_iter)):
#     s = s0 # always initalize an episode in s0
#     q_sa = Q1[s0,1]
#
#     for t in range(timeout):
#         # epsilon-greedily select an action
#         a1 = e_greedy(Q1, epsilon(T))
#         a2 = e_greedy(Q2, epsilon(T))
#
#         # action is defined jointly by the two agents
#         a = A_joint[a1,a2]
#
#         # query transition model to obtain s'
#         s_prime = transition(s, a)
#
#         # query the reward model to obtain r
#         r1 = R[s_prime, 0]
#         r2 = R[s_prime, 1]
#
#         # update Q
#         #Q1[s,a1] = (1 - alpha(T)) * Q1[s,a1] + alpha(T) * ((1. - gamma) * r1 + gamma * Q1[s_prime,:].max())
#         #Q2[s,a2] = (1 - alpha(T)) * Q2[s,a2] + alpha(T) * ((1. - gamma) * r1 + gamma * Q2[s_prime,:].max())
#         Q1[s,a1] = Q1[s,a1] + alpha(T) * (r1 + gamma * Q1[s_prime, :].max() - Q1[s,a1])
#         Q2[s,a2] = Q2[s,a2] + alpha(T) * (r2 + gamma * Q2[s_prime, :].max() - Q2[s,a2])
#         # update s
#         s = s_prime
#
#         # terminate when a goal is made
#         if r1 != 0 or r2 != 0: break
#
#     ERR.append(np.abs(Q1[s0,1] - q_sa))
#     A.append(alpha(T))
# #
# #
# # N = len(ERR)
# # X = range(0, N, 500)
# # ERR = ERR[0:N:500]
#
# N = len(ERR)
# X = range(0,N,500)
# ERR = ERR[0:N:500]
#
# plt.figure()
# plt.plot(A)
# # plt.ylim((0, .5))
# plt.title('alpha value for Q-Learning')
# plt.xlabel('Simulation Iteration')
#
# plt.show()
#
#
# plt.plot(X, ERR, 'red')
# plt.ylim((0,.5))
# plt.title('Q-Learning')
# plt.xlabel('Simulation Iteration')
# plt.ylabel('Q-Value Difference')
# # plt.xticks(np.arange(0, len(ERR), 250000))
# plt.show()
#
# quit()

#################################################################################
####foe-Q
def epi_greedy(pi, s, esilon):
    if np.random.rand() >= esilon:
        return np.random.choice(pi.shape[1])
    else:
        # print "pi[s,:]: ", pi[s,:]
        return np.random.choice(range(pi.shape[1]), p=pi[s,:])
# # epsilon greedy action selector
## Use PULP for LP:
from pulp import LpVariable, LpProblem, LpMinimize, LpMaximize

# ===========================================================
def pi_solve(pi, Q, s):
    # The 'prob' variable will contain the problem data.
    prob = LpProblem('FindPi', LpMinimize)

    a0 = LpVariable('a0', 0.0)  # the minimum is 0.0
    a1 = LpVariable('a1', 0.0)  # the minimum is 0.0
    a2 = LpVariable('a2', 0.0)  # the minimum is 0.0
    a3 = LpVariable('a3', 0.0)  # the minimum is 0.0
    a4 = LpVariable('a4', 0.0)  # the minimum is 0.0
    v  = LpVariable('v' , 0.0)

    # The objective function is added to 'prob' first
    prob += v, "to minimize"

    # constraints
    prob +=  a0 * Q[s,0,0] + a1 * Q[s,1,0] + a2 * Q[s,2,0] + a3 * Q[s,3,0] + a4 * Q[s,4,0] <= v, 'constraint 1'
    prob +=  a0 * Q[s,0,1] + a1 * Q[s,1,1] + a2 * Q[s,2,1] + a3 * Q[s,3,1] + a4 * Q[s,4,1] <= v, 'constraint 2'
    prob +=  a0 * Q[s,0,2] + a1 * Q[s,1,2] + a2 * Q[s,2,2] + a3 * Q[s,3,2] + a4 * Q[s,4,2] <= v, 'constraint 3'
    prob +=  a0 * Q[s,0,3] + a1 * Q[s,1,3] + a2 * Q[s,2,3] + a3 * Q[s,3,3] + a4 * Q[s,4,3] <= v, 'constraint 4'
    prob +=  a0 * Q[s,0,4] + a1 * Q[s,1,4] + a2 * Q[s,2,4] + a3 * Q[s,3,4] + a4 * Q[s,4,4] <= v, 'constraint 5'

    prob +=  1.0*a0 + 1.0*a1 + 1.0*a2 + 1.0*a3 + 1.0*a4 == 1, 'constraint 6'

    prob.solve()

    pi_prime = [a.varValue for a in prob.variables()[:5]]
    # print pi_prime

    # aa = 1

    return np.array(pi_prime)

#
# # from pulp import LpVariable, LpProblem, LpMinimize, LpMaximize
# def pi_solve(pi, Q, s):
#     # The 'prob' variable will contain the problem data.
#     prob = LpProblem('FindPi', LpMaximize)
#
#     a0 = LpVariable('a0', 0.0)  # the minimum is 0.0
#     a1 = LpVariable('a1', 0.0)  # the minimum is 0.0
#     a2 = LpVariable('a2', 0.0)  # the minimum is 0.0
#     a3 = LpVariable('a3', 0.0)  # the minimum is 0.0
#     a4 = LpVariable('a4', 0.0)  # the minimum is 0.0
#     v  = LpVariable('v' , 0.0)
#
#     # The objective function is added to 'prob' first
#     prob += a0 * Q[s,0,0] + a1 * Q[s,1,0] + a2 * Q[s,2,0] + a3 * Q[s,3,0] + a4 * Q[s,4,0] + \
#             a0 * Q[s,0,1] + a1 * Q[s,1,1] + a2 * Q[s,2,1] + a3 * Q[s,3,1] + a4 * Q[s,4,1] + \
#             a0 * Q[s,0,2] + a1 * Q[s,1,2] + a2 * Q[s,2,2] + a3 * Q[s,3,2] + a4 * Q[s,4,2] + \
#             a0 * Q[s,0,3] + a1 * Q[s,1,3] + a2 * Q[s,2,3] + a3 * Q[s,3,3] + a4 * Q[s,4,3] + \
#             a0 * Q[s,0,4] + a1 * Q[s,1,4] + a2 * Q[s,2,4] + a3 * Q[s,3,4] + a4 * Q[s,4,4], "to maximize"
#
#     # constraints
#     prob +=  a0 * Q[s,0,0] + a1 * Q[s,1,0] + a2 * Q[s,2,0] + a3 * Q[s,3,0] + a4 * Q[s,4,0] >= 0, 'constraint 1'
#     prob +=  a0 * Q[s,0,1] + a1 * Q[s,1,1] + a2 * Q[s,2,1] + a3 * Q[s,3,1] + a4 * Q[s,4,1] >= 0, 'constraint 2'
#     prob +=  a0 * Q[s,0,2] + a1 * Q[s,1,2] + a2 * Q[s,2,2] + a3 * Q[s,3,2] + a4 * Q[s,4,2] >= 0, 'constraint 3'
#     prob +=  a0 * Q[s,0,3] + a1 * Q[s,1,3] + a2 * Q[s,2,3] + a3 * Q[s,3,3] + a4 * Q[s,4,3] >= 0, 'constraint 4'
#     prob +=  a0 * Q[s,0,4] + a1 * Q[s,1,4] + a2 * Q[s,2,4] + a3 * Q[s,3,4] + a4 * Q[s,4,4] >= 0, 'constraint 5'
#
#     prob +=  1.0*a0 + 1.0*a1 + 1.0*a2 + 1.0*a3 + 1.0*a4 == 1, 'constraint 6'
#
#     prob.solve()
#
#     pi_prime = [a.varValue for a in prob.variables()[:5]]
#     # print "pi_prime: ", pi_prime
#
#     aa = 1
#     pp = np.array(pi_prime)
#     pp = pp /np.sum(pp)
#
#     for p in pi_prime:
#         if not p >=0 or np.isnan(p):
#             return pi.copy()
#     if abs(np.sum(pp) -1) > 0.001:
#         return pi.copy()
#     return pp



n_iter = 50000 # numer of iterations of Q learning
timeout = 10 # episode length timeout threshold, shoud never occur
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
# print pi_1
# print pi_2

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
for T in (range(n_iter)):
    if T%100 == 0:
        print "Itr: ", T
    s = s0 # always initalize an episode in s0
    q_sa = Q1[s0,1,4] #player A moves south, player B sticks

    for t in range(timeout):
        # epsilon-greedily select an action
        a1 = epi_greedy(pi_1, s, epsilon(T))
        a2 = epi_greedy(pi_2, s, epsilon(T))

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

        # # use LP to find pi
        # pp1 = pi_solve(pi_1, Q1, s)
        # pp2 = pi_solve(pi_2, Q2, s)
        # pi_1[s,:] = pp1
        # pi_2[s,:] = pp2
        pi_1[s,:] = pi_solve(pi_1[s,:], Q1, s)
        pi_2[s,:] = pi_solve(pi_2[s,:], Q2, s)

        # update V
        V1[s] = np.min(Q1[s,:,:].sum(axis=0) * pi_1[s,a1])
        V1[s] = np.min(Q2[s,:,:].sum(axis=0) * pi_2[s,a2])

        # update s
        s = s_prime

        # terminate when a goal is made
        if r1 != 0 or r2 != 0:
            break
        # if r1 == 100 or r2 == 100:
        #     break

    ERR.append(np.abs(Q1[s0,1,4] - q_sa)) #player A moves south, player B sticks

print ERR
plt.plot(ERR,'r')
plt.ylim((0,.5))
plt.title('Foe-Q')
plt.xlabel('Simulation Iteration')
plt.ylabel('Q-Value Difference')
# plt.xticks(np.arange(0, len(ERR), 100))
plt.show()

#np.save("fQErr1.npy", ERR)
f=open('foeq.txt', 'w')

quit()














#################################################################################
####CEQ
####foe-Q
def epi_greedy(pi, s, esilon):
    if np.random.rand() >= esilon:
        return np.random.choice(pi.shape[1])
    else:
        return np.random.choice(range(pi.shape[1]), p=pi[s,:])

from pulp import LpVariable, LpProblem, LpMinimize

def pi_solve(pi, Q, s):
    # The 'prob' variable will contain the problem data.
    prob = LpProblem('FindPi', LpMinimize)

    a0 = LpVariable('a0', 0.0)  # the minimum is 0.0
    a1 = LpVariable('a1', 0.0)  # the minimum is 0.0
    a2 = LpVariable('a2', 0.0)  # the minimum is 0.0
    a3 = LpVariable('a3', 0.0)  # the minimum is 0.0
    a4 = LpVariable('a4', 0.0)  # the minimum is 0.0
    v = LpVariable('v', 0.0)

    # The objective function is added to 'prob' first
    prob += v, "to minimize"

    # constraints
    prob +=  a0 * Q[s,0,0] + a1 * Q[s,1,0] + a2 * Q[s,2,0] + a3 * Q[s,3,0] + a4 * Q[s,4,0] <= v, 'constraint 1'
    prob +=  a0 * Q[s,0,1] + a1 * Q[s,1,1] + a2 * Q[s,2,1] + a3 * Q[s,3,1] + a4 * Q[s,4,1] <= v, 'constraint 2'
    prob +=  a0 * Q[s,0,2] + a1 * Q[s,1,2] + a2 * Q[s,2,2] + a3 * Q[s,3,2] + a4 * Q[s,4,2] <= v, 'constraint 3'
    prob +=  a0 * Q[s,0,3] + a1 * Q[s,1,3] + a2 * Q[s,2,3] + a3 * Q[s,3,3] + a4 * Q[s,4,3] <= v, 'constraint 4'
    prob +=  a0 * Q[s,0,4] + a1 * Q[s,1,4] + a2 * Q[s,2,4] + a3 * Q[s,3,4] + a4 * Q[s,4,4] <= v, 'constraint 5'

    prob +=  1.0*a0 + 1.0*a1 + 1.0*a2 + 1.0*a3 + 1.0*a4 == 1, 'constraint 6'

    prob.solve()

    pi_prime = [a.varValue for a in prob.variables()[:5]]

    return np.array(pi_prime)


n_iter = 1000 # numer of iterations of Q learning
timeout = 10 # episode length timeout threshold, shoud never occur

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

for T in tqdm(range(n_iter)):
    s = s0 # always initalize an episode in s0
    q_sa = Q1[s0,1,4] #player A moves south, player B sticks

    for t in range(timeout):
        # epsilon-greedily select an action
        a1 = epi_greedy(pi_1, s, epsilon(T))
        a2 = epi_greedy(pi_2, s, epsilon(T))

        # action is defined jointly by the two agents
        a = A_joint[a1,a2]

        # query transition model to obtain s'
        s_prime = transition(s, a)

        # query the reward model to obtain r
        r1 = R[s_prime, 0]
        r2 = R[s_prime, 1]

        # # update Q
        Q1[s,a1,a2] = (1 - alpha(T)) * Q1[s,a1,a2] + alpha(T) * (r1 + gamma * V1[s_prime])
        Q2[s,a2,a1] = (1 - alpha(T)) * Q2[s,a2,a1] + alpha(T) * (r2 + gamma * V2[s_prime])


        # use LP to find pi
        pi_1[s,:] = pi_solve(pi_1, Q1, s)
        pi_2[s,:] = pi_solve(pi_2, Q2, s)

        # update V
        V1[s] = np.min(Q1[s,:,:].sum(axis=0) * pi_1[s,a1])
        V1[s] = np.min(Q2[s,:,:].sum(axis=0) * pi_2[s,a2])

        # update s
        s = s_prime

        # terminate when a goal is made
        if r1 == 100 or r2 == 100: break

    ERR.append(np.abs(Q1[s0,1,4] - q_sa)) #player A moves south, player B sticks

print '----------------'
print ERR
plt.plot(ERR)
plt.ylim((0,.5))
plt.title('Correlated-Q')
plt.xlabel('Simulation Iteration')
plt.ylabel('Q-Value Difference')
# plt.xticks(np.arange(0, len(ERR), 100))
f=open('ceq.txt', 'w')

plt.show()





