# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Crocodile3'
__mtime__ = '2018/11/19'
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
from sklearn.datasets import load_boston


# 第一步收集数据
boston = load_boston()
X = boston.data
y = boston.target
# print(boston.feature_names)

#第二步，模型训练

from sklearn.model_selection import train_test_split
# 将数据划分为训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)

import time
from sklearn.linear_model import LinearRegression


def native_linear():
    model = LinearRegression()
    start = time.clock()
    model.fit(X_train,y_train,y_train)
    train_score = model.score(X_train,y_train)
    cv_score = model.score(X_test,y_test)
    print("elsape:{0:.6f};train_score:{1:0.6f};cv_score:{2:.6f}".format(time.clock()-start,train_score,cv_score))


# 模型优化
# 数据归一化处理

model = LinearRegression(normalize=True)
# 增加多项式特征
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.pipeline import Pipeline

def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features",polynomial_features),('linear_regression',linear_regression)])
    return pipeline

# 使用二阶多项式来拟合
model = polynomial_model(degree=2)
start = time.clock()
model.fit(X_train,y_train)

train_score = model.score(X_train,y_train)
cv_score = model.score(X_test,y_test)
print("elsape:{0:.6f};train_score:{1:0.6f};cv_score:{2:.6f}".format(time.clock()-start,train_score,cv_score))

# 绘制学习曲线
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import numpy as np

def plot_learning_curve(plt, estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
cv = ShuffleSplit(n_splits=10,test_size=0.2,random_state=0)
plt.figure(figsize=(18,4),dpi=200)
title = 'Learning Curves (degree={0})'
degrees = [1,2,3]
start = time.clock()
plt.figure(figsize=(18,4),dpi=200)
for i in range(len(degrees)):
    plt.subplot(1,3,i+1)
    plot_learning_curve(plt,polynomial_model(degrees[i]),title.format(degrees[i]),X,y,ylim=(0.01,1.01),cv=cv)
    
print('elaspe:{0:.6f}'.format(time.clock()-start))