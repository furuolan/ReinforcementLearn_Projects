import matplotlib.pyplot as plt
import argparse
import pickle

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parse pkl')
    parser.add_argument('pkl', metavar='p', type=str, nargs=1)

    args = vars(parser.parse_args())

    filename = args['pkl'][0]

    with open(filename, 'r') as infile:
        ERR = pickle.load(infile)

    plt.plot(ERR)
    plt.ylim((0,.5))
    plt.title('Foe-Q')
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-Value Difference')

    plt.show()