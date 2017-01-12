import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import *
import time
import data_utils
import random_walk
from linear_td_lamda import Linear_TD

######################################################################
### (1) Generate random walk training sequences (100 training sets with 10 sequences each)
seedFac = 10
walks = random_walk.generate_walks(1000, seedFac)
training_folds = random_walk.split_training_sets(walks, 10)

### (2) The ideal predictions for the 7 state random-walk are the weights we want to learn,
# that are true probabilities of the walk terminating at state G.
# Note that these values are external to the learning problem (as they requre full knowledge of the MDP),
# and are only used to assess the model.
ideal_predictions = np.array(np.linspace(1./6., 5./6., 5), dtype=np.float64)
rmse = lambda x, y: np.sqrt(np.mean((x-y)**2), dtype = np.float64)


# ## (3) Extracted into numeric csv files from the plots in the paper using webplotdigitizer
# ## (http://arohatgi.info/WebPlotDigitizer/)
fig3 = data_utils.load_fig('fig3.csv')
fig4_lambdas, fig4 = data_utils.load_fig4('fig4')
fig5 = data_utils.load_fig('fig5.csv')


# ## (4) Experiment 1 (Sutton88 figure 3)
# # ex1_lambda = [0.,.1,.3,.5,.7,.9,1.]
# ex1_lambda= np.array([0.,.1,.3,.5,.7,.9,1.], dtype=np.float64)
# ex1_rmse = []
# ex1_sigma = []
# alpha = .025  ### larger aloha learns faster, but if it too large, overshoot.
# epsilon = .03     ### smaller, converge better, but will overfit if too small.
# # epsilon = .01
#
# # E =[0.1, 0.07, 0.05, 0.03,0.01, 0.001, 0.0001]
#
# print "alpha = ", alpha
# print "eps = ", epsilon
#
# start = time.time()
# for L in ex1_lambda:
#     L_rmse = []
#     for f in range(len(training_folds)):
#         td = Linear_TD(lam = L, learning_rate=alpha, epsilon=epsilon)
#         td.fit(training_folds[f])
#         L_rmse.append(rmse(ideal_predictions, td.w[1:6]))
#     ex1_rmse.append(np.mean(L_rmse))
#     ex1_sigma.append(np.std(L_rmse))
# end = time.time()
# ex1_sigma /= np.sqrt(len(training_folds))
#
# print "eps = ", epsilon, '>>>>>>>>'
# print ex1_rmse
# print "time: ", end - start
# print 'ex1_sigma:', ex1_sigma
# print '-----------------'
#
#
#
# plt.plot(ex1_lambda, ex1_rmse, 'go-')
# # plt.errorbar(ex1_lambda, ex1_rmse, yerr=ex1_sigma, fmt='--o', ecolor='g')
# # plt.title('a='+str(alpha) + ', eps=' + str(epsilon) +
# #           'Average Error of Random Walk Problem Under Repeated Presentations', fontsize='10')
# plt.ylabel('RMS Error', fontsize='13')
# plt.xlabel('$\lambda$', fontsize='13')
# plt.grid()
# plt.xlim(-0.1,1.1)
# plt.show()
# quit()

###############################################
### (5) Experiment 2
ex2a_alpha = np.linspace(0., .6, 13)
ex2a_lambda = np.array(fig4_lambdas).astype(np.float)
ex2a_rmse = []
epsilon = .03
T=[]

for L in ex2a_lambda:
    alphas = []
    for a in ex2a_alpha:
        rmses = []
        for f in range(len(training_folds)):
            td = Linear_TD(lam = L, learning_rate=a, epsilon=epsilon, incremental_updates=True)
            td.fit(training_folds[f])
            rmses.append(rmse(ideal_predictions, td.w[1:6]))
        alphas.append(np.mean(rmses))
    ex2a_rmse.append(alphas)

ex2a_points = np.asarray(ex2a_rmse).T
max_rmse = .7
ex2a_points[ex2a_points > max_rmse] = np.nan
#
# plt.plot(ex2a_alpha, ex2a_points,  'o-')
# plt.ylabel('RMS Error', fontsize='12')
# plt.xlabel(r'$\alpha$', fontsize='20')
# plt.title("seedFac: " + str(seedFac))
# plt.grid()
# plt.xlim(-0.1,0.7)
# legend = ['$\lambda$=' + str(l) for l in fig4_lambdas]
# plt.legend(legend, loc='best')
# plt.show()



###############################################
### (6) Experiment 2-2
#iterate over all<lambda,alpha> and select the best alpha for each lambda ex2b_lambda = np.linspace(0., 1., 11)
ex2b_lambda = np.linspace(0., 1., 11)
ex2b_alpha = np.linspace(0., .6, 13)
optimal_params = []

for i in tqdm(range(len(ex2b_lambda))):
    L = ex2b_lambda[i]
    best_alpha = -1.
    lowest_err = float('inf')
    for a in ex2b_alpha:
        rmses = []
        for f in training_folds:
            td = Linear_TD(lam = L, learning_rate=a, incremental_updates=True)
            td.fit(f)
            rmses.append(rmse(ideal_predictions, td.w[1:6]))
        avg_err = np.mean(rmses)
        if avg_err < lowest_err:
            best_alpha = np.asscalar(a)
            lowest_err = avg_err
    optimal_params.append((L, best_alpha))
optimal_params = np.round(optimal_params, decimals=2).tolist()



##### Now we can finally run using the optimal parameter tuples!
# epsilon = .05
ex2b_rmse = []
ex2b_sigma = []
for L, a in optimal_params:
    rmses = []
    for f in tqdm(range(len(training_folds))):
        td = Linear_TD(lam = L, learning_rate=a, incremental_updates=True)
        td.fit(training_folds[f])
        rmses.append(rmse(ideal_predictions, td.w[1:6]))
    ex2b_rmse.append(np.mean(rmses))
    ex2b_sigma.append(np.std(rmses))
ex2b_sigma /= np.sqrt(len(training_folds))

plt.plot(ex2b_lambda, ex2b_rmse, 'go-')
# plt.errorbar(ex2b_lambda, ex2b_rmse, yerr=ex2b_sigma, fmt='--o', ecolor='g')
plt.title('Average Error at Best Alpha on Random Walk Problem', fontsize='10')
plt.ylabel('RMS Error', fontsize='13')
plt.xlabel('$\lambda$', fontsize='13')
plt.grid()
plt.xlim(-0.1,1.1)
plt.show()






