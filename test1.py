# test file
import sys
import numpy as np
import matplotlib.pyplot as plt

xx = np.arange(100)
yy = np.sqrt(xx)
yy2 = xx**2
fig, ax = plt.subplots()

ax.plot(xx, yy)
ax.plot(xx, yy2)
plt.show()