#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   KNN.py    
@Contact :   1094404954@qq.com
@License :   (C)Copyright Cheng Hoiyuen

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/24 1:26   郑海远      1.0         None
'''

# import lib
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

#测试
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

with open(r"../data/all.csv") as file:
    dataset = pd.read_csv(file)
    inputs = ["field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z",
              "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z"]
    outputs = ["left_uwb_thread"] #"left_uwb_thread", "right_uwb_thread"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch = 50
seclect_data = int(0.7 * dataset.shape[0])
x_train = torch.tensor(dataset[inputs].iloc[:seclect_data, :].values).to(torch.float32)
y_train = torch.tensor(dataset[outputs].iloc[:seclect_data, :].values*10).to(torch.float32)

x_test = torch.tensor(dataset[inputs].iloc[seclect_data:, :].values).to(torch.float32)
y_test = torch.tensor(dataset[outputs].iloc[seclect_data:, :].values*10).to(torch.float32)



knn = KNeighborsClassifier(n_neighbors=3,weights='uniform',algorithm='auto')

knn.fit(x_train,y_train)
# 传入测试数据，做预测
y_pred = knn.predict(x_test)

# 求准确率
accuracy = accuracy_score(y_test,y_pred)
print('预测准确率：',accuracy)
'''预测准确率： 0.9555555555555556'''

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
estimator=KNeighborsClassifier()

##确定网格交叉验证的参数
param_dict=[{
    'weights':["uniform"],
          'n_neighbors':[i for i in range (1,20)]},
    {
    'weights': ['distance'],
    'n_neighbors': [i for i in range(1, 20)],
    'p': [i for i in range(1, 20)]
    }]

estimator=GridSearchCV(estimator,param_grid=param_dict,cv=3)
estimator.fit(x_train,y_train)

##预测
y_predict=estimator.predict(x_test)
##模型评估


##打印最佳参数
print("最佳参数：\n",estimator.best_params_)
##最佳结果
print("最佳结果：\n",estimator.best_score_)
##最佳预估器
print("最佳预估器：\n",estimator.best_estimator_)

knn_clf = estimator.best_estimator_
print(knn_clf.predict(x_test).reshape(-1, 1).shape,y_test.shape)
knn_clf.score(knn_clf.predict(x_test).reshape(-1, 1), y_test.numpy())
