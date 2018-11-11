# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Crocodile3'
__mtime__ = '2018/11/11'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

import pandas as pd
import numpy as np
import codecs
import networkx as nx
import matplotlib.pyplot as plt


# 读取数据并获取姓名
data = pd.read_csv('data.csv',encoding='gbk')
name = []
for n in data['姓名']:
    name.append(n)

a = np.zeros([2,3])
word_vector = np.zeros([len(name),len(name)])

# 计算学院共线矩阵
i = 0
while i < len(name):
    academy1 = data['学院'][i]
    j = i +1
    while j <len(name):
        academy2 = data['学院'][j]
        if academy1 == academy2:
            word_vector[i][j] += 1
            word_vector[j][i] += 1
        j = j +1
    i = i + 1
# print(word_vector)
np_data = np.array(word_vector)
pd_data = pd.DataFrame(np_data)
pd_data.to_csv('result.csv')


# 共线矩阵计算

words = codecs.open("word_node.txt", "a+", "utf-8")
i = 0
while i<len(name):  #len(name)
    student1 = name[i]
    j = i + 1
    while j<len(name):
        student2 = name[j]
        #判断学生是否共现 共现词频不为0则加入
        if word_vector[i][j]>0:
            words.write(student1 + " " + student2 + " "
                + str(word_vector[i][j]) + "\r\n")
        j = j + 1
    i = i + 1

a = []
f = codecs.open('word_node.txt', 'r', 'utf-8')
line = f.readline()

i = 0
A = []
B = []
while line != "":
    a.append(line.split())  # 保存文件是以空格分离的

    A.append(a[i][0])
    B.append(a[i][1])
    i = i + 1
    line = f.readline()
elem_dic = tuple(zip(A, B))

f.close()

import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'

colors = ["red", "green", "blue", "yellow"]
G = nx.Graph()
G.add_edges_from(list(elem_dic))
# nx.draw(G,with_labels=True,pos=nx.random_layout(G),font_size=12,node_size=2000,node_color=colors) #alpha=0.3
pos=nx.spring_layout(G,iterations=50)
# pos = nx.random_layout(G)
nx.draw_networkx_nodes(G, pos, alpha=0.2, node_size=1200, node_color=colors)
nx.draw_networkx_edges(G, pos, node_color='r', alpha=0.3)  # style='dashed'
nx.draw_networkx_labels(G, pos, font_family='sans-serif', alpha=0.5)  # font_size=5
plt.show()