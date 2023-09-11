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
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston



from sklearnex import patch_sklearn,unpatch_sklearn
import joblib
from sklearn.model_selection import GridSearchCV

patch_sklearn()

def make_predict(model,df,inputs,name):
    y_pred = model.predict(df[inputs])
    pd.DataFrame({name:y_pred}).to_csv("../data/corrected.csv")

def train(df,inputs,output):
    print("Training "+output)
    X = df[inputs].values
    y = df[output].values

    # XGBoost训练过程
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, verbosity=1, objective='reg:gamma')
    model.fit(X_train, y_train)

    # 对测试集进行预测
    y_test_pred = model.predict(X_test)



    print('MSE  test: %.3f' % (
        mean_squared_error(y_test, y_test_pred)))
    print('R^2 test: %.3f' % (
        r2_score(y_test, y_test_pred)))
    importances = list(model.feature_importances_)


    # Saving feature names for later use
    feature_list = list(df[inputs].columns)
    # print(feature_list)
    feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Print out the feature and importances
    # print(feature_importances)
    x_values = list(range(len(importances)))
    x_values = [str(x) for x in x_values]
    # print(x_values)
    # Make a bar chart
    fig, ax = plt.subplots()
    ax.bar(x_values, importances, orientation='vertical')
    # Tick labels for x axis
    # ax.set_xticks(x_values, feature_list)
    ax.set_xticks(x_values)  # 设置刻度
    ax.set_xticklabels(feature_list, rotation=30)
    # Axis labels and title
    ax.set_ylabel('Importance')
    ax.set_xlabel('Variable')
    ax.set_title('Variable Importances '+output)
    ax.legend()
    model.save_model("../models/XGB"+output)
    return model

    # def ceate_feature_map(features):
    #     outfile = open(output+'.fmap', 'w')
    #     i = 0
    #     for feat in features:
    #         outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    #         i = i + 1
    #     outfile.close()
    #
    # ceate_feature_map(df[inputs].columns)  # 特征名列表
    #
    # fig, ax = plt.subplots()
    # fig.set_size_inches(60, 60)
    # model.get_booster().feature_names = feature_list
    # xgb.plot_tree(model,ax = ax,fmap=output+'.fmap')
    # fig.savefig(output+'.jpg')


    # fig, ax = plt.subplots()
    # rf = RandomForestRegressor(n_estimators=1)
    # mse_trains, mse_tests = [], []
    # for iter in range(50):
    #     rf.fit(X_train, y_train)
    #     y_train_predicted = rf.predict(X_train)
    #     y_test_predicted = rf.predict(X_test)
    #     # mse_train = mean_squared_error(y_train, y_train_predicted)
    #     # mse_test = mean_squared_error(y_test, y_test_predicted)
    #     # print("Iteration: {} Train mse: {} Test mse: {} Depth: {} N: {}".format(iter, mse_train, mse_test,rf.max_depth, rf.n_estimators))
    #     Train_r2 = r2_score(y_train, y_train_predicted)
    #     Test_r2 = r2_score(y_test, y_test_predicted)
    #     # print('R^2 train: %.3f, test: %.3f' % (
    #     #         r2_score(y_train, y_train_predicted),
    #     #         r2_score(y_test, y_test_predicted)))
    #     rf.n_estimators += 1
    #     mse_trains.append(Train_r2)
    #     mse_tests.append(Test_r2)
    # ax.plot(np.arange(0,len(mse_trains)),mse_trains,label="mse_train"+output)
    # ax.plot(np.arange(0, len(mse_tests)), mse_tests, label="mse_test"+output)
    # print(np.max(mse_trains),np.max(mse_tests))
    # ax.legend()






with open(r"../data/all.csv") as file:
    dataset = pd.read_csv(file)
    inputs = ["field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z",
              "field.linear_acceleration.x", "field.linear_acceleration.y","field.linear_acceleration.z"]
    outputs = ["right_uwb_thread","left_uwb_thread"]
    # correted = ["corrected_right_uwb", "corrected_left_uwb"]
    model = train(dataset,inputs,outputs[0])

    # make_predict(model)
    train(dataset,inputs, outputs[1])
    plt.show()

    # Training right_uwb_thread
    # MSE train: 0.003, test: 0.018
    # R ^ 2 train: 0.961, test: 0.800


    # Training left_uwb_thread
    # MSE train: 0.004, test: 0.023
    # R ^ 2  train: 0.976, test: 0.837

