import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# arr = np.random.randn(100)
# data = pd.Series(arr)
# print(data[:10])
# print(data.skew())
# print(data.kurt())
# import matplotlib.pyplot as plt
#
# data.hist(bins=100)
# plt.show()

points_1 = np.random.randint(1, 7, 10000)
points_2 = np.random.randint(1, 7, 10000)
points_3 = np.random.randint(1, 7, 10000)

data = pd.Series(points_1 + points_2 + points_3)
print(data[:10])

rs = data.value_counts().sort_index(ascending=True)
print('rs: ', rs)

print('Æ«¶Èskew: ', data.skew())
print('·å¶Èkurt:', data.kurt())

fig = plt.figure()

ax1 = fig.add_subplot(221)
plt.hist2d(data, data, bins=9,)

ax2 = fig.add_subplot(222)
plt.hist(data, bins=10, color="c", histtype='bar')

ax3 = fig.add_subplot(223)
plt.hist(data, bins=11, histtype='step')

ax4 = fig.add_subplot(224)
plt.hist(data, bins=12, histtype='step')


plt.show()