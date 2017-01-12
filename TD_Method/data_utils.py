import numpy as np
import os


def load_fig(filename):
    data = np.loadtxt(filename, delimiter=',')[:,1]
    return data


def load_fig4(fig_path):
    lambda_list = []
    pt_list = []
    for f in os.listdir(fig_path):
        if f.endswith(".csv"):
            lambda_val = int(f.split('_')[1].split('.')[0])
            lambda_val /= 10. if lambda_val > 1 else 1
            lambda_list.append(lambda_val)

            full_path = os.path.join(fig_path, f)
            pt_list.append(load_fig(full_path))

    pt_array = np.vstack(pt_list).T

    return (lambda_list, pt_array)

