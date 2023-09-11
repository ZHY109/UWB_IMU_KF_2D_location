#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   simulator.py
@Contact :   1094404954@qq.com
@License :   (C)Copyright Cheng Hoiyuen

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/3/28 15:24   郑海远      1.0         None
'''

# import lib
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.metrics import mean_squared_error
import seaborn as sn
import xgboost as xgb
import gif

pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 130  # 图形分辨率
sn.set_theme(style='darkgrid')  # 图形主题


t = 0.1
duration = 30
def gaussian_distribution_generator(var):
    return np.random.normal(loc=0.0, scale=var, size=None)

def points_MSE(ax,ay,bx,by):
    sum = 0
    for i in range(len(ax)):
        sum+=(ax[i]-bx[i])**2+(ay[i]-by[i])**2
    return np.sum(sum)/len(ax)

def RMSE(ax,ay,bx,by):
    return math.sqrt(points_MSE(ax,ay,bx,by))

#sgn函数
def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(new_df,threshold):
    data = new_df
    data = data.values.T.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('sym6')
    maxlevel = pywt.dwt_max_level(len(data), w.dec_len)
    signal = pywt.wavedec(data, w, level=maxlevel)

    abs_cd1 = np.abs(np.array(signal[-1]))
    median_cd1 = np.median(abs_cd1)

    usecoeffs = []
    usecoeffs.append(signal[0])

    print(maxlevel)
    # for s in signal[1:]:
    #     length= len(s)
    #     for k in range(length):
    #         if (abs(s[k]) >= lamda / np.log2(maxlevel-i)):
    #             s[k] = sgn(s[k]) * (abs(s[k]) - lamda / np.log2(maxlevel-i))
    #         else:
    #             s[k] = 0.0
    #     i+=1
    #     usecoeffs.append(s)
    for i in range(1, len(signal)):
        usecoeffs.append(pywt.threshold(signal[i], threshold * max(signal[i])) ) # 将噪声滤波

    recoeffs = pywt.waverec(usecoeffs, w)#信号重构
    return recoeffs



# 状态转移矩阵，上一时刻的状态转移到当前时刻
A = np.array([[1,0,t,0,t*t/2,0],
              [0,1,0,t,0,t*t/2],
              [0,0,1,0,t,0],
              [0,0,0,1,0,t],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]])

# 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性
Q = 0.1*np.array([[1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]])

# Q = np.array([[1,0,0,0,0,0],
#               [0,1,0,0,0,0],
#               [0,0,1,0,0,0],
#               [0,0,0,1,0,0],
#               [0,0,0,0,1,0],
#               [0,0,0,0,0,1]])

# 观测噪声协方差矩阵R，p(v)~N(0,R)
# R = np.array([[1,0,0,0],
#               [0,1,0,0],
#               [0,0,1,0],
#               [0,0,0,1]])
R = np.array([[0.6,0,0,0],
              [0,0.6,0,0],
              [0,0,0.1,0],
              [0,0,0,0.1]])
# 状态观测矩阵
H = np.array([[1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]])

# 控制输入矩阵B
B = None
# 初始化
X0 = np.array([[0],
               [0.8],
               [0],
               [0],
               [0],
               [0]])

# 状态估计协方差矩阵P初始化
P = np.identity(6)


@gif.frame
def plot(i):
    fig, axs = plt.subplots(figsize=(6, 6))
    j = copy(i)
    if i>17:
        i=i-17

    axs.plot(data['x'][:j], data['y'][:j], ".", label="geo", linewidth=1)
    axs.plot(X_filtered[:i], Y_filtered[:i], "-.", label="filter+geo", linewidth=1)
    axs.plot(position_posterior_x_est[:i], position_posterior_y_est[:i], ":", label="filter+geo+imu",linewidth=2)

    axs.plot(real_x[:j], real_y[:j], label="groundtruth")
    axs.set_xlabel('m')  # Add an x-label to the axes.
    axs.set_ylabel('m')  # Add an x-label to the axes.
    axs.set_title("location")
    axs.set_xticks([-2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4])
    axs.set_yticks([0, 0.8, 1.6, 2.4, 3.2, 4])
    axs.legend()  # Add a legend.

if __name__ == "__main__":
    # ---------------------------初始化-------------------------
    with open(r"../data/2023_4_26/all.csv") as file:
        data = pd.read_csv(file)
        # X_filtered = wavelet_noising(data['x'], 1.1)
        # Y_filtered = wavelet_noising(data['y'], 1.1)
        X_filtered = data['corrected_x']
        Y_filtered = data['corrected_y']
        imu_x = data["field.linear_acceleration.x"]
        imu_y = data["field.linear_acceleration.y"]
        l = len(X_filtered)
    real_x = []
    real_y = []
    b = 30
    for i in range(b):
        real_x.append(0)
        real_y.append(0.8)
    a = 40
    # a = 22
    for i in range(a):
        real_x.append(0)
        if (i/a*3.2+0.8<3.2):real_y.append(i/a*3.2+0.8)
        else:real_y.append(3.2)
    for i in range(l-a-b):
        real_x.append(i/(l-a-b)*-1.6)
        real_y.append(3.2)
    # for i in range(l-90):
    #     real_x.append(-1.6)
    #     real_y.append(3.2)
    # print(X_filtered,Y_filtered)
    X_true = np.array(X0)  # 真实状态初始化
    X_posterior = np.array(X0)
    P_posterior = np.array(P)

    position_y_true = []
    position_x_true = []

    position_y_measure = []
    position_x_measure = []

    position_y_prior_est = []
    position_x_prior_est = []

    position_posterior_y_est = []
    position_posterior_x_est = []

    frames = []
    for i in range(l):
        # -----------------------生成真实值----------------------

        X_true = [[X_filtered[i]],[Y_filtered[i]],[0],[0],[0],[0]]



        Z_measure = [[X_filtered[i]],[Y_filtered[i]],[imu_x[i]],[imu_y[i]]]
        position_x_measure.append(Z_measure[0])
        position_y_measure.append(Z_measure[1])
        # ----------------------进行先验估计---------------------
        X_prior = np.dot(A, X_posterior)
        position_x_prior_est.append(X_prior[0])
        position_y_prior_est.append(X_prior[1])
        # 计算状态估计协方差矩阵P
        P_prior_1 = np.dot(A, P_posterior)
        P_prior = np.dot(P_prior_1, A.T) + Q
        # ----------------------计算卡尔曼增益,用numpy一步一步计算Prior and posterior
        k1 = np.dot(P_prior, H.T)
        k2 = np.dot(np.dot(H, P_prior), H.T) + R
        K = np.dot(k1, np.linalg.inv(k2))
        # ---------------------后验估计------------
        # print(Z_measure,np.dot(H, X_prior))
        X_posterior_1 = Z_measure - np.dot(H, X_prior)
        X_posterior = X_prior + np.dot(K, X_posterior_1)
        position_posterior_x_est.append(X_posterior[0][0])
        position_posterior_y_est.append(X_posterior[1][0])

        # 更新状态估计协方差矩阵P
        P_posterior_1 = np.eye(6) - np.dot(K, H)
        P_posterior = np.dot(P_posterior_1, P_prior)

        frame = plot(i)
        frames.append(frame)
        print(len(frames))
        # 根据帧序列frames,动画持续时间duration，生成gif动画
    gif.save(frames, 'example.gif', duration=3.5)


    # Geo_imu_x = []
    # Geo_imu_y = []
    # X_posterior = np.array(X0)
    # P_posterior = np.array(P)
    # for i in range(l):
    #     # -----------------------生成真实值----------------------
    #
    #     X_true = [[data['x'][i]],[data['y'][i]],[0],[0],[0],[0]]
    #     position_x_true.append(X_true[0])
    #     position_y_true.append(X_true[1])
    #
    #
    #     Z_measure = [[data['x'][i]],[data['y'][i]],[imu_x[i]],[imu_y[i]]]
    #     position_x_measure.append(Z_measure[0])
    #     position_y_measure.append(Z_measure[1])
    #     # ----------------------进行先验估计---------------------
    #     X_prior = np.dot(A, X_posterior)
    #     position_x_prior_est.append(X_prior[0])
    #     position_y_prior_est.append(X_prior[1])
    #     # 计算状态估计协方差矩阵P
    #     P_prior_1 = np.dot(A, P_posterior)
    #     P_prior = np.dot(P_prior_1, A.T) + Q
    #     # ----------------------计算卡尔曼增益,用numpy一步一步计算Prior and posterior
    #     k1 = np.dot(P_prior, H.T)
    #     k2 = np.dot(np.dot(H, P_prior), H.T) + R
    #     K = np.dot(k1, np.linalg.inv(k2))
    #     # ---------------------后验估计------------
    #     # print(Z_measure,np.dot(H, X_prior))
    #     X_posterior_1 = Z_measure - np.dot(H, X_prior)
    #     X_posterior = X_prior + np.dot(K, X_posterior_1)
    #     Geo_imu_x.append(X_posterior[0][0])
    #     Geo_imu_y.append(X_posterior[1][0])
    #
    #     # 更新状态估计协方差矩阵P
    #     P_posterior_1 = np.eye(6) - np.dot(K, H)
    #     P_posterior = np.dot(P_posterior_1, P_prior)

    # 可视化显示
    if True:
        # fig, axs = plt.subplots()
        # # axs.plot(position_x_true,position_y_true, "-", label="true", linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(0, len(X_filtered)), X_filtered, "-",label="Filted_loc_x",  linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(0, len(data['x'])), data['x'], "-",label="Raw_loc_x",
        #          linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(0, len(data['x'])), real_x, "-", label="real",
        #          linewidth=1)  # Plot some data on the axes.
        # axs.set_ylabel('m')  # Add an x-label to the axes.
        # axs.set_title("X location")
        # axs.set_xlabel('points sequence')  # Add an x-label to the axes.
        # axs.legend()  # Add a legend.
        #
        # fig, axs = plt.subplots()
        # # axs.plot(position_x_true,position_y_true, "-", label="true", linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(0, len(Y_filtered)), Y_filtered, "-",label="dynamic",  linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(0, len(data['y'])), data['y'], "-",label="raw",
        #          linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(0, len(data['y'])), real_y, "-", label="real",
        #          linewidth=1)  # Plot some data on the axes.
        # axs.set_ylabel('m')  # Add an x-label to the axes.
        # axs.set_xlabel('points sequence')  # Add an x-label to the axes.
        # axs.set_title("Y location")
        # axs.legend()  # Add a legend.


        # fig, axs = plt.subplots(figsize=(6, 6))
        # axs.plot(np.arange(len(real_x)), real_x, ".", label="groundtruth", linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(len(real_x)), data['x'], ".", label="geo", linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(len(X_filtered)), X_filtered, ".", label="filter+geo",
        #          linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(len(X_filtered)), position_posterior_x_est, ".", label="filter+geo+imu", linewidth=1)  # Plot some data on the axes.
        # axs.set_title("x")
        # axs.legend()  # Add a legend.
        #
        # fig, axs = plt.subplots(figsize=(6, 6))
        # axs.plot(np.arange(len(real_x)), real_y, ".", label="groundtruth", linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(len(real_x)), data['y'], ".", label="geo", linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(len(X_filtered)), Y_filtered, ".", label="filter+geo",
        #          linewidth=1)  # Plot some data on the axes.
        # axs.plot(np.arange(len(X_filtered)), position_posterior_y_est, ".", label="filter+geo+imu",
        #          linewidth=1)  # Plot some data on the axes.
        # axs.set_title("y")
        # axs.legend()  # Add a legend.

        # fig, axs = plt.subplots(figsize=(6,6))
        # axs.plot(data['x'], data['y'], ".", label="geo", linewidth=1)  # Plot some data on the axes.
        # axs.plot(X_filtered, Y_filtered, "-.", label="filter+geo", linewidth=1)  # Plot some data on the axes.
        # axs.plot(position_posterior_x_est, position_posterior_y_est, ":", label="filter+geo+imu",
        #          linewidth=2)  # Plot some data on the axes.
        # # axs.plot(Geo_imu_x, Geo_imu_y, ":", label="geo+imu",
        # #          linewidth=1)  # Plot some data on the axes.
        # axs.plot(real_x, real_y, label="groundtruth")
        # axs.set_xlabel('m')  # Add an x-label to the axes.
        # axs.set_ylabel('m')  # Add an x-label to the axes.
        # axs.set_title("location")
        # axs.set_xticks([-2.4,-1.6,-0.8,0,0.8,1.6,2.4])
        # axs.set_yticks([0,0.8,1.6,2.4,3.2,4])
        # axs.legend()  # Add a legend.
        #
        # df = {'algo':[],'x':[],'y':[]}
        # for i in range(len(data['x'])):
        #     df['algo'].append("goundtruth")
        #     df['x'].append(real_x[i])
        #     df['y'].append(real_y[i])
        # for i in range(len(data['x'])):
        #     df['algo'].append("geo")
        #     df['x'].append(data['x'][i])
        #     df['y'].append(data['y'][i])
        # for i in range(len(X_filtered)):
        #     df['algo'].append("dynamic+geo")
        #     df['x'].append(X_filtered[i])
        #     df['y'].append(Y_filtered[i])
        # for i in range(len(position_posterior_x_est)):
        #     df['algo'].append("dynamic+fussion")
        #     df['x'].append(position_posterior_x_est[i])
        #     df['y'].append(position_posterior_y_est[i])


        # df = pd.DataFrame(df)
        # # print(df)
        # fig, axs = plt.subplots(figsize=(6, 6))
        # p = sn.scatterplot(data=df,x='x',y='y',hue='algo',style='algo')
        # p.set_xlabel("m")
        # p.set_ylabel("m")
        # p.set_xticks([-2.4, -1.6, -0.8, 0, 0.8, 1.6, 2.4])
        # p.set_yticks([0, 0.8, 1.6, 2.4, 3.2, 4])
        #
        #
        # print("True&geo:", points_MSE(real_x, real_y, data['x'], data['y']))
        # print("True&dynamic+geo:", points_MSE(real_x, real_y, X_filtered, Y_filtered),
        #       "per:",(points_MSE(real_x, real_y, X_filtered, Y_filtered)-
        #               points_MSE(real_x, real_y, data['x'], data['y']))/points_MSE(real_x, real_y, data['x'], data['y']))
        # print("True&dynamic+fussion:", points_MSE(real_x, real_y, position_posterior_x_est, position_posterior_y_est),
        #       "per:", (points_MSE(real_x, real_y, position_posterior_x_est, position_posterior_y_est) -
        #                points_MSE(real_x, real_y, data['x'], data['y'])) / points_MSE(real_x, real_y, data['x'],
        #                                                                               data['y']))
        #
        # print("True&geo:", RMSE(real_x, real_y, data['x'], data['y']))
        # print("True&dynamic+geo:", RMSE(real_x, real_y, X_filtered, Y_filtered),
        #       "per:", (RMSE(real_x, real_y, X_filtered, Y_filtered) -
        #                RMSE(real_x, real_y, data['x'], data['y'])) / RMSE(real_x, real_y, data['x'],
        #                                                                               data['y']))
        # print("True&dynamic+fussion:", RMSE(real_x, real_y, position_posterior_x_est, position_posterior_y_est),
        #       "per:", (RMSE(real_x, real_y, position_posterior_x_est, position_posterior_y_est) -
        #                RMSE(real_x, real_y, data['x'], data['y'])) / RMSE(real_x, real_y, data['x'],
        #                                                                               data['y']))
        # # print("True&Measure:", points_MSE(position_x_true, position_y_true, position_x_measure, position_y_measure))
        # # print("True&KF:", points_MSE(position_x_true, position_y_true, position_x_prior_est, position_y_prior_est))
        plt.show()