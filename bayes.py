# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'Crocodile3'
__mtime__ = '2018/11/9'
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
import re
from math import log
from pprint import pprint

import feedparser
from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]    #1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    """
    创建一个不重复的词汇列表，
    返回一个不重复的词汇列表
    :param dataSet:
    :return:
    """
    vocabSet = set([])
    # 循环所有文本内容
    for document in dataSet:
        # 利用集合对每条留言内容的词汇进行去重，然后再与词汇集合求并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList,inputSet):
    """
    判断词汇表中的单词是否在对应的留言条中出现，
    并生成一个对应词汇长度（32）的列表（1：出现，0：没出现）
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        # 判断留言条中的单词是否在词汇表中
        if word in vocabList:
            # 如果在，就根据该词汇在列表中索引位置的值设为1
            returnVec[vocabList.index(word)] = 1
        else:
            # 否则，则输出该词汇不在词汇表中
            print('the word :{} is not in my Vocabulary!'.format(word))
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    """
    构造朴素贝叶斯分类器
    :param trainMatrix: 训练文本的数字向量集，每一行表示一条留言
    :param trainCategory: 训练文本属性的向量集
    :return:
    """
    # print(trainMat)
    # print(trainCategory)
    numTrainDocs = len(trainMatrix)  # 训练向量集的长度，即有多少条留言样本
    numWords = len(trainMatrix[0])   # 第一个样本集向量第一个元素的长度，即该条留言样本在词汇表中是否出现构成的向量
    pAbusive = sum(trainCategory)/float(numTrainDocs)   # 计算出侮辱性文章的先验概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]  # 表示每一行的向量集
            p0Denom += sum(trainMatrix[i])  # 词汇表中的词在每一行出现的总次数
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive


    
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # P(w|c1) * P(c1)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    # P(w|c0) * P(c0)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    # 1. 加载数据集
    listOPosts, listClasses = loadDataSet()
    # 2. 创建单词集合
    myVocabList = createVocabList(listOPosts)
    # 3. 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in listOPosts:
        # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 4. 训练数据
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    # 5. 测试数据
    testEntry = ['love', 'my', 'dalmation','stupid']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    
    
def textParse(bigString):
    """
    将邮件内容按照单词进行切分
    :param bigString: 邮件的内容
    :return: 每封邮件的内容以单词为单位的列表
    """
    listOfTokens = re.split(r'\W.*?', bigString)
    # print(listOfTokens)
    # 将单词转化为小写并且摒弃长度小于2的单词
    return [tok.lower() for tok in listOfTokens if len(tok)>2]


def spamTest():
    """
    对贝叶斯垃圾邮件分类器进行自动化处理
    """
    docList = []   # 以每个单词为维度
    classList = []  # 以每封邮件的内容为维度
    fullText  = []  #
    for i in range(1,26):
        # 读取邮件内容，并将其转换为单词列表
        wordList = textParse(open('email/spam/%d.txt'%i).read())
        
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList =textParse(open('email/ham/%d.txt' % i).read())
        # print(wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
    # 创建词汇列表
    vocabList = createVocabList(docList)
    # 生成一个0-49的列表的训练集
    # 划分训练集和测试集，从训练集中随机取出10个测试集，将剩余的40个作为训练集
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        # 随机从训练集中取出一个数加入到测试集中
        testSet.append(trainingSet[randIndex])
        # 将加入到测试集中元素从训练集中删除
        del(trainingSet[randIndex])
    # 构造并生成训练向量和训练的类型列表
    trainMat = []
    trainClass = []
    # 遍历训练集
    for docIndex in trainingSet:
        # 从邮件内容列表中取出邮件内容，并将邮件内容转为为向量加到训练向量集中
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        # 取出邮件对应的类标签加到训练类型列表中
        trainClass.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array((trainClass)))
    errorCount = 0
    # 遍历测试集，然后根据贝叶斯模型分类计算出分类的错误率
    for docIndex in testSet:
        # 取出测试集中邮件的内容然后转为向量列表
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the erroe rate is :{}'.format(float(errorCount)/len(testSet)))



# 个人广告分类器
import operator
def calcMostFreq(vocabList,fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]


def localWords(feed1,feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
    vocabList = createVocabList(docList)
    top30words = calcMostFreq(vocabList,fullText)
    for pairW  in top30words:
        if pairW[0] in vocabList:vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is :{}'.format(float(errorCount)/len(testSet)))
    return vocabList,p0V,p1V
    
    
def getTopWords(ny,sf):
    """
    最具表征性的词汇显示
    :param ny:
    :param sf:
    :return:
    """
    vocabList,p0V,p1V = localWords(ny,sf)
    topNy = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:
            topNy.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key=lambda  pair: pair[1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNy,key=lambda pair: pair[1],reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])
    


if __name__ == '__main__':
    # postingList, classVec = loadDataSet()
    # myVocabList = createVocabList(postingList)
    # print(myVocabList)
    # print(postingList)
    # 将第二条留言转为向量
    # returnVec = setOfWords2Vec(myVocabList,postingList[1])
    # print(returnVec)
    # 创建一个训练向量列表，每个元素代表一条留言的向量
    # trainMat = []
    # 遍历所有的留言
    # for postinDoc in postingList:
        # 将每条留言对的文字转为向量
        # trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print(trainMat)
    # p0V, p1V, pAb = trainNB0(trainMat, classVec)
    # print(p0V)
    # print(p1V)
    # print(pAb)
    # testingNB()
    # spamTest()
    ny = feedparser.parse('http://newyork.craiglist.org/stp/index.rss')
    print(ny['entries'])
    # sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # print(sf)
    # vocabList,pSF,pNY = localWords(ny,sf)
    #
