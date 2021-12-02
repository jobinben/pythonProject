import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dict={
    'a': ['A', 'B', 'C', 'D', 'E'],
    'b': [90, 85, 90, 80, 95],
    'c': [60, 65, 70, 65, 75]
}
data = pd.DataFrame(dict)
print(data)
# x=[1, 2, 3, 4, 5]
# y=data['b']
#
# plt.plot(x, y, 'ro--')
# plt.show()

# ����ͼ
s = pd.Series([12, 13, 15, 11, 14])
max = s.max()
max_num = s.loc[s==max].index[0]
min = s.min()
min_num = s.loc[s==min].index[0]
# x=s.index
# y=s
# plt.plot(x, y, 'go--')
# plt.axvline(max_num, color='r', linestyle=':')
# plt.axhline(s[max_num]-1, color='b', linestyle=':')
# plt.show()

# ����ͼ
# x=np.arange(1,6)
# y=[90, 5, 10, 50, 95]
# plt.bar(x, y, width=0.5, color='r', edgecolor='b')
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.title('�����ͳ��')
# plt.show()

# ����ͼ
y=np.arange(1,6)
width = [90, 5, 10, 50, 95]
height = 0.4
plt.barh(y, width,height, color='r', edgecolor='b')
plt.rcParams['font.sans-serif']=['SimHei']
plt.title('�����ͳ��')
plt.show()


