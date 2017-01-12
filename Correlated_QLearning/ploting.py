
import numpy as np
import matplotlib.pyplot as plt


with open("tryfoe.txt") as f:
    data = f.read()

plt.plot(data)
plt.ylim((0,.5))
plt.title('Correlated-Q')
plt.xlabel('Simulation Iteration')
plt.ylabel('Q-Value Difference')
# plt.xticks(np.arange(0, len(ERR), 100))
plt.show()
