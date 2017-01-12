from collections import namedtuple
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt

State = namedtuple('State', 'ball_pos alpha beta')

# functional definition of model
def transition(s, a):
    '''s in S is a 3-tuple comprising the current state, a in A is an action vector
    containing the actions selected by A and B'''
    # unpack variables
    ball_pos, alpha, beta = s

    # impose boundary constraint (can't fall off the edge)
    if (alpha + a[0] > 7 or alpha + a[0] < 0):
        a[0] = 0
    if (beta + a[1] > 7 or beta + a[1] < 0):
        a[1] = 0

    # init ball_pos_prime
    bp_prime = ball_pos

    # impose collision dynamics
    if (alpha + a[0]) == beta + a[1]:
        # choose who goes first
        first_player = np.random.randint(2)
        second_player = 1 - first_player

        # handle the case where the second player is sticking
        if a[second_player] == 0:
            a = [0,0]

        # for all other acitons, only the second player is blocked
        else:
            a[second_player] = 0

        # change possesion when second player has the ball
        bp_prime = 1 - ball_pos if ball_pos == second_player else ball_pos

    alpha_prime = alpha + a[0]
    beta_prime = beta + a[1]

    return State(bp_prime, alpha_prime, beta_prime)

def get_grid(S):
    grid = np.zeros((8,))
    grid[[0,3,4,7]] = 5
    grid[S.alpha] = 3 if S.ball_pos == 0 else 1
    grid[S.beta] = 4 if S.ball_pos == 1 else 2
    return grid.reshape(2,4)


def plot_state(S):
    grid = get_grid(S)

    # make a color map of fixed colors
    cmap = ListedColormap(['white','blue','red','green','magenta','yellow'])
    bounds=[0,1,2,3,4,5,6]
    norm = BoundaryNorm(bounds, cmap.N)

    # tell imshow about color map so that only set colors are used
    img = plt.imshow(grid,interpolation='nearest',
                        cmap = cmap,norm=norm)

    # make a color bar
    plt.colorbar(img,cmap=cmap,
                    norm=norm,boundaries=bounds,ticks=[0,1,2,3,4,5,6])


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

