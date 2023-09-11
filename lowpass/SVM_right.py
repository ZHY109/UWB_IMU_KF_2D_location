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

import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import numpy as np
import pandas as pd

#测试
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

with open(r"../data/all.csv") as file:
    dataset = pd.read_csv(file)
    inputs = ["field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z",
              "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z"]
    outputs = ["right_uwb_thread"] #"left_uwb_thread", "right_uwb_thread"

batch = 50
x = dataset[inputs].values
y = np.ravel(dataset[outputs].values*10)


from sklearn.svm import SVC,SVR
from sklearn.preprocessing import StandardScaler #可做标准化
from sklearn.model_selection import train_test_split

#读入数据
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


#svm建模
# svm_classification =SVC()
# svm_classification.fit(x_train,y_train)
# #模型效果
# print(svm_classification.score(x_test,y_test))
from sklearnex import patch_sklearn,unpatch_sklearn
import joblib
from sklearn.model_selection import GridSearchCV
#定义参数的组合
params={
'kernel':('rbf','poly'),
'C':[1000,5000,10000,50000,100000],
'degree':[2,3,4,5],
'coef0':[0.0,3.0,5.0 ]
}
patch_sklearn()
#unpapatch_sklearn()

svm_classification = SVC(C=100000, coef0= 0.0, degree=5, kernel='poly')
model = GridSearchCV(SVC, param_grid=params, cv=10)

# model = GridSearchCV(svm_classification, param_grid=params, cv=10)
scores = cross_val_score(svm_classification, x, y, cv=10)
print(scores.mean())

# print("最好的参数组合: ", model.best_params_)
# print("最好的score: ", model.best_score_)
#最好的参数组合:  {'C': 100000, 'coef0': 0.0, 'degree': 5, 'kernel': 'poly', 'probability': False}
# 最好的score:  0.899469387755102
# joblib.dump(model.best_estimator_, './models/right_uwb_bestSVC.pkl')

# model = joblib.load('./models/right_uwb_bestSVC.pkl')
