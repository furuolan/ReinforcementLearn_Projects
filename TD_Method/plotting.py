import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import *
import time
import data_utils
import random_walk
from linear_td_lamda import Linear_TD

####################################
E =[0.1, 0.07, 0.05, 0.03, 0.01, 0.001, 0.0001]

x1=[0.19927633670754047, 0.19710594354042466, 0.18465126226023085, 0.15528553291142821, 0.13894710547450567, 0.14022846191551802, 0.16991036929418954]
x2=[0.15497234561122883, 0.14549855149703217, 0.13174932068320333, 0.11931666936894601, 0.11587412645645878, 0.13405635801419805, 0.16866742012791786]
x3=[0.11439943499258973, 0.10969104269190366, 0.10239971290036957, 0.099909705689562284, 0.10597071677784684, 0.13238004529144692, 0.16884773340600454]
x4=[0.085762223284012715, 0.086006240211584817, 0.086956205853476368, 0.092977140810976661, 0.10614683189008156, 0.1351753709034007, 0.17109044166180579]
x5=[0.093741395322895327, 0.09584322948045082, 0.10089963349119058, 0.10831140765257503, 0.12005889227924692, 0.14362429015558786, 0.17688054096556305]
x6=[0.12036350132032883, 0.12078080459293922, 0.12225999372994428, 0.12511209040200899, 0.13133692626240639, 0.14978460849696776, 0.18116351375756157]
x7=[0.12454281423384911, 0.12451075029676169, 0.12509783505938274, 0.12709238462616437, 0.13255589414598387, 0.15037982695498814, 0.18161202885889297]

y =[0.,.1,.3,.5,.7,.9,1.]

plt.plot(y, x1)
plt.plot(y, x2)
plt.plot(y, x3)
plt.plot(y, x4, 'o-', linewidth=2.5)
plt.plot(y, x5)
plt.plot(y, x6)
plt.plot(y, x7)

plt.ylabel('RMS Error', fontsize='13')
plt.xlabel('$\lambda$', fontsize='13')

legend = ['$\epsilon$ = '+str(e) for e in E]
plt.legend(legend, loc='lower right', fontsize='small')
plt.grid()
plt.xlim(-0.1,1.1)
plt.show()


# E =[0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# x = [1, 2, 3, 4, 5, 6]
# times = [1.60672712326,3.03853297234,21.5899209976,51.3018498421,100.380096912,133.179901838]
# labels = [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001]
#
# plt.plot(x, times, 'c',linewidth=3)
# # You can specify a rotation for the tick labels in degrees or with keywords.
# plt.xticks(x, labels, rotation='vertical')
# # Pad margins so that markers don't get clipped by the axes
# plt.margins(0.2)
# plt.ylabel('Time',fontsize='13')
# plt.xlabel("$\epsilon$", fontsize='23')
# # Tweak spacing to prevent clipping of tick-labels
# plt.subplots_adjust(bottom=0.15)
# plt.grid()
# plt.show()




