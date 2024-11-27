import numpy as np
import matplotlib.pyplot as plt

def I(V,Is,alpha): return Is*(np.exp(alpha*V)-1)

x = np.linspace(0,4,100)
y = I(x,0.005,1.3)

plt.plot(x,y)

plt.show()