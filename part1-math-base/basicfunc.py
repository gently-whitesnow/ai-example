import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import ndarray
# %matplotlib inline

from typing import Callable
from typing import Dict

from funcs import square, leaky_relu



print(square(10))
print(leaky_relu(-10))

### graf

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(12, 6))  # 2 Rows, 1 Col

input_range = np.arange(-2, 2, 0.01)
ax[0].plot(input_range, square(input_range))
ax[0].plot(input_range, square(input_range))
ax[0].set_title('Square function')
ax[0].set_xlabel('input')
ax[0].set_ylabel('input')

ax[1].plot(input_range, leaky_relu(input_range))
ax[1].plot(input_range, leaky_relu(input_range))
ax[1].set_title('"ReLU" function')
ax[1].set_xlabel('input')
ax[1].set_ylabel('output')
plt.show()
