# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 08:49:16 2021

@author: Administrator
"""

# 导入 pandas 包
import pandas as pd
#使用 read_csv 从 film.txt 文件中读取数据，并用分号隔开
df= pd.read_csv('film.txt', delimiter=';')
# 筛选指定内容
df=df[['上映时间','闭映时间', '票房/万']]
 # 对齐
pd.set_option('display.unicode.east_asian_width', True)
print(df.head())
# 数据清洗：去除带有 NaN（空值）的数据行
df=df.dropna()
# 将上映时间和闭映时间转换为时间类型
df['上映时间'] = pd.to_datetime(df['上映时间'])
df['闭映时间'] = pd.to_datetime(df['闭映时间'])
# 计算电影放映天数
df['放映天数']=(df['闭映时间'] - df['上映时间']).dt.days + 1
# 将票房数据转换为浮点型
df['票房/万'] = df['票房/万'].astype(float)
# 计算日均票房
df['日均票房/万'] = df['票房/万']/df['放映天数']
# 重置索引列，不添加新的列
df = df.reset_index(drop=True)
# 对齐
pd.set_option('display.unicode.east_asian_width', True)

# 输出从文件中读取的部分结果
print(df.head())

# 导入 sklearn 包
from sklearn import linear_model
# 设定 x 和 y 的值
x = df[['放映天数']]
y = df[['日均票房/万']]
# 初始化线性回归模型
regr = linear_model.LinearRegression()
# 线性回归拟合（训练）
regr.fit(x, y)

print(regr)

# 导入画图包
import matplotlib.pyplot as plt
# 设置中文字体为 SimHei，简黑字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示的问题
plt.rcParams['axes.unicode_minus'] = False
# 可视化
# 设置标题
plt.title('放映天数与票房关系图（一元线性回归分析）')
# 设置 x，y 轴的标题，x 轴显示的值为放映天数。
plt.xlabel('放映天数')
plt.ylabel('日均票房收入\万元')
# 画散点图
plt.scatter(x, y, color='black')
# 画出预测点，预测点的宽度为 1，颜色为红色
plt.scatter(x, regr.predict(x), color='red',linewidth=1, marker='*')
# 添加图例
plt.legend(['原始值','预测值'], loc = 2)
# 显示图像
plt.show()


# 导入包：引入模型选择模块的 train_test_split
from sklearn.model_selection import train_test_split
# 拆分训练集和测试集
# 调用接口：指定训练集与测试集的大小，返回的训练集与测试集切分结果
# train_size:训练样本占比，若为 None 时，自动设置为 0.75；
# test_size: 测试样本占比，若为 None 时，自动设置为 0.25
# random_state:随机数的种子
x_train, x_test,y_train, y_test=train_test_split(df[['放映天数']],df[['日均票房/万\
']], train_size=0.8, test_size = 0.2, random_state = 1)
# 建立线性回归模型
regr = linear_model.LinearRegression()
# 使用训练集进行拟合
regr.fit(x_train, y_train)
# 给出测试集的预测结果
y_pred = regr.predict(x_test)
#print(y_pred)
plt.title(u'预测值与实际值比较（一元线性回归）')
plt.ylabel(u'日均票房收入\万元')
plt.plot(range(len(y_pred)),y_pred,'red', linewidth=2.5,label=u"预测值")
plt.plot(range(len(y_test)),y_test,'green',label=u"实际值")
plt.legend(loc=2)
#显示预测值与测试值曲线
plt.show()









