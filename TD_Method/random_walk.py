import numpy as np

# np.random.seed(0)
def bounded_random_walk(limit=100, init_state=3, absorbing_states=[0,6], seed=0):
    np.random.seed(seed)
    walk = [init_state]
    for i in range(limit):
        next_state = walk[-1]+1 if np.random.rand()>0.5 else walk[-1]-1
        walk.append(next_state)
        if next_state in absorbing_states:
            break
    return walk


def encode_one_hot(v, num_states=7):
    return np.eye(num_states)[v]


def generate_walks(N, seedFac, seed=0):
    walks = []
    for s in range(N):
        walk = bounded_random_walk(seed = seedFac*s+seed)

        # np.random.seed(seeee)
        # seeee = np.random.randint(0, 100000, 1)
        # print seeee
        # walk = bounded_random_walk(seeee)
        walk_vector = encode_one_hot(walk)
        # walks.append(walk_vector)
        walks.append(np.array(walk_vector, dtype=np.float64))
    return walks


def split_training_sets(walks, n):
    n = max(1, n)
    return [walks[i:i+n] for i in range(0, len(walks), n)]
