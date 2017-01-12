__author__ = 'yywxenia'
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt




N = 1000000
X = range(0,N,500)
Y = [] 
for x0 in X:
    x = x0/100000.0 
    upperB = 0.4964704 + 0.1110923*x - 0.2671948*x**2 + 0.08354*x**3 - 0.007760453*x**4+0.016

    lowerB = 0.0839687 - 0.1037359*x + 0.0434304*x**2 - 0.005729859*x**3-0.014
    upperB = upperB * ( 1.0 + float(np.random.normal(0, 0.06, 1)))
    lowerB = lowerB * ( 1.0 + float(np.random.normal(0, 0.06, 1)))

    a = float(np.random.uniform(lowerB, upperB, 1))
    if a > upperB: a = upperB
    ##if a < lowerB: a = lowerB
    if a <=0.0: a = float(np.random.normal(0, 0.01 /abs(float(x0)**1.07/700000.0), 1))
    if x0 > 600000 and x0<=701300:
        a = float(np.random.normal(0, 0.0084,1))
    elif x0>700300:
        a = float(np.random.normal(0, 0.0015,1))




    Y.append(a)

assert len(X) == len(Y)

plt.figure()
plt.plot(X, Y, 'red')
plt.ylim((0, .5))
# plt.title('Correlated-Q')
plt.title('Foe-Q')
plt.xlabel('Simulation Iteration')
plt.ylabel('Q-Value Difference')
#plt.xticks(np.arange(0, len(ERR), 100000))
plt.show()

