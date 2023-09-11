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
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import math

t = 0.1
duration = 30
def gaussian_distribution_generator(var):
    return np.random.normal(loc=0.0, scale=var, size=None)

def points_MSE(ax,ay,bx,by):
    sum = 0
    for i in range(len(ax)):
        sum+=math.sqrt((ax[i]-bx[i])**2+(ay[i]-by[i])**2)
    return np.sum(sum)/len(ax)

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
R = np.array([[0.25,0,0,0],
              [0,0.25,0,0],
              [0,0,1,0],
              [0,0,0,1]])
# 状态观测矩阵
H = np.array([[1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]])

# 控制输入矩阵B
B = None
# 初始化
X0 = np.array([[1],
               [1],
               [0],
               [0],
               [0],
               [0]])

# 状态估计协方差矩阵P初始化
P = np.identity(6)




if __name__ == "__main__":
    # ---------------------------初始化-------------------------
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

    for i in range(duration):
        # -----------------------生成真实值----------------------
        # 生成过程噪声
        w = np.array([[gaussian_distribution_generator(Q[0, 0])],
                      [gaussian_distribution_generator(Q[1, 1])],
                      [gaussian_distribution_generator(Q[2, 2])],
                      [gaussian_distribution_generator(Q[3, 3])],
                      [gaussian_distribution_generator(Q[4, 4])],
                      [gaussian_distribution_generator(Q[5, 5])],
                      ])
        X_true = np.dot(A, X_true) + w  # 得到当前时刻状态
        position_x_true.append(X_true[0, 0])
        position_y_true.append(X_true[1, 0])
        # -----------------------生成观测值----------------------
        # 生成观测噪声
        v = np.array([[gaussian_distribution_generator(R[0, 0])],
                      [gaussian_distribution_generator(R[1, 1])],
                      [gaussian_distribution_generator(R[2, 2])],
                      [gaussian_distribution_generator(R[3, 3])],
                      ])
        Z_measure = np.dot(H, X_true) + v  # 生成观测值,H为单位阵E
        position_x_measure.append(Z_measure[0, 0])
        position_y_measure.append(Z_measure[1, 0])
        # ----------------------进行先验估计---------------------
        X_prior = np.dot(A, X_posterior)
        position_x_prior_est.append(X_prior[0, 0])
        position_y_prior_est.append(X_prior[1, 0])
        # 计算状态估计协方差矩阵P
        P_prior_1 = np.dot(A, P_posterior)
        P_prior = np.dot(P_prior_1, A.T) + Q
        # ----------------------计算卡尔曼增益,用numpy一步一步计算Prior and posterior
        k1 = np.dot(P_prior, H.T)
        k2 = np.dot(np.dot(H, P_prior), H.T) + R
        K = np.dot(k1, np.linalg.inv(k2))
        # ---------------------后验估计------------
        X_posterior_1 = Z_measure - np.dot(H, X_prior)
        X_posterior = X_prior + np.dot(K, X_posterior_1)
        position_posterior_x_est.append(X_posterior[0, 0])
        position_posterior_y_est.append(X_posterior[1, 0])
        # 更新状态估计协方差矩阵P
        P_posterior_1 = np.eye(6) - np.dot(K, H)
        P_posterior = np.dot(P_posterior_1, P_prior)

    # 可视化显示
    if True:
        fig, axs = plt.subplots()
        axs.plot(position_x_true,position_y_true, "-", label="true", linewidth=1)  # Plot some data on the axes.
        axs.plot(position_x_measure,position_y_measure, "-", label="measure", linewidth=1)  # Plot some data on the axes.
       # axs.plot(position_x_prior_est,position_y_prior_est, "-", label="position_prior_est", linewidth=1)  # Plot some data on the axes.
        axs.plot(position_posterior_x_est,position_posterior_y_est, "-", label="KF",
                    linewidth=1)  # Plot some data on the axes.
        axs.set_title("location")
        axs.set_xlabel('m')  # Add an x-label to the axes.
        axs.set_ylabel('m')  # Add an x-label to the axes.
        axs.legend()  # Add a legend.

        print("True&Measure:", points_MSE(position_x_true, position_y_true, position_x_measure, position_y_measure))
        print("True&KF:", points_MSE(position_x_true, position_y_true, position_x_prior_est, position_y_prior_est))
        plt.show()