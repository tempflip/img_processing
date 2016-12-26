import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 


pdf = stats.norm(0, 10).pdf

print pdf

lin = np.linspace(-5, 5, 100)

a = np.array([[pdf(x) * pdf(y) for x in lin] for y in lin])

print a


plt.imshow(a, interpolation="none")
plt.show()


