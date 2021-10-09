import pandas as pd
import numpy as np

keys = ['a', 'b', 'c']
indexs = ['001', '002', '003']

data = pd.DataFrame(np.arange(1,10).reshape(3,3), columns=keys, index=indexs)
data['dateTime'] = '2021-10-09'
dateTime = data['dateTime'].str.split('-')[0]

# 分离年月日
timeData = {'year': dateTime[0], 'month': dateTime[1], 'day': dateTime[2]}
timeKeys = ['year', 'month', 'day']
for i in timeKeys:
    data[i] = timeData[i]


# 导入数据
filepath = './supermarket.xlsx'
data2_1 = pd.read_excel(filepath,sheetname=0) # 读取
pd.set_option('display.unicode.east_asian_width', True) # 对齐
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 操作数据

data2_1['支付方式'] = '银行转账'
data2_1['销售金额'] = data2_1['单价'] * data2_1['数量']
user = data2_1['客户'].str.split('-')[0]
data2_1['客户姓名'] = user[0]
data2_1['客户ID'] = user[1]

print(data2_1.head(3))

