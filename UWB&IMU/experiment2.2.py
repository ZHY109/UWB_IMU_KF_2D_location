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
import pywt
import torch
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



def plot(i):
    fig, axs = plt.subplots(figsize=(8, 8))
    j = copy(i)

    axs.plot(geo_x[:j], geo_y[:j], ".", label="geo", linewidth=1)
    axs.plot(X_filtered[:i], Y_filtered[:i], "-.", label="filter+geo", linewidth=1)
    axs.plot(position_posterior_x_est[:i], position_posterior_y_est[:i], ":", label="filter+geo+imu",linewidth=2)

    axs.plot(real_x[:j], real_y[:j],label="groundtruth",color='red')
    # print(real_x,real_y)
    axs.set_xlabel('m')  # Add an x-label to the axes.
    axs.set_ylabel('m')  # Add an x-label to the axes.
    axs.set_title("location")
    axs.set_xticks([-4.8,-3.6, -2.4,-1.2, 0,1.2, 2.4,3.6, 4.8])
    axs.set_yticks([-4.8,-3.6, -2.4,-1.2, 0,1.2, 2.4,3.6, 4.8])

    axs.legend()  # Add a legend.
    plt.legend(loc='lower left')
    plt.savefig('experiment2.2.png')
    plt.show()

import numpy as np

def signaltonoise(signal, noise):
    signal_power = np.sum(signal ** 2) / len(signal)
    noise_power = np.sum(noise ** 2) / len(noise)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr



def add_noise(data, noise_level,bias):
    noise = np.random.normal(bias, noise_level, len(data))
    return noise



if __name__ == "__main__":

    real_x = []
    real_y = []


    import numpy as np

    real_x = [3.2 for i in range(20)]
    real_y = [4-i/5*0.6 for i in range(0,15)]
    real_y.extend( [4 - i / 5 for i in range(15, 20)])

    theta = np.linspace(0, -np.pi-np.pi/2, 100)
    r = 3.2

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    real_x.extend(x)
    real_y.extend(y)
    print(real_y)
    # geo_x = copy(real_x)
    # geo_y = copy(real_y)
    # geo_x = add_noise(geo_x,)

    position_posterior_x_est = copy(real_x)
    position_posterior_y_est = copy(real_y)


    position_posterior_x_est = position_posterior_x_est + add_noise(position_posterior_x_est,0.5,-0.5)
    position_posterior_y_est = position_posterior_y_est + add_noise(position_posterior_y_est, 0.2, 0)
    ppx_noise =  add_noise(position_posterior_x_est,0.3,-0.5)
    ppy_noise =  add_noise(position_posterior_y_est, 0.1, 0)

    geo_x = position_posterior_x_est+  add_noise(position_posterior_x_est,0.2,0)
    geo_y = position_posterior_y_est + add_noise (position_posterior_y_est,0.2,0)
    gx_noise = add_noise(position_posterior_x_est,0.2,0) + ppx_noise
    gy_noise = add_noise (position_posterior_y_est,0.2,0) + ppy_noise

    print(f" geo_x loactioning(MSE)："
          f"{points_MSE(real_x, real_y, geo_x,geo_y)},"
          f"snr：{signaltonoise(geo_x,gx_noise), signaltonoise(geo_y,gy_noise)}")
    print(RMSE(real_x, real_y, geo_x,geo_y))
    X_filtered = copy(position_posterior_x_est)
    Y_filtered = copy(position_posterior_y_est)
    X_filtered = wavelet_noising(X_filtered,0.5)
    Y_filtered = wavelet_noising(Y_filtered, 0.4)
    print(f" dynamic filtering loactioning(MSE)："
          f"{points_MSE(real_x, real_y, X_filtered,Y_filtered)},"
          f"snr：{signaltonoise(X_filtered,ppx_noise), signaltonoise(Y_filtered,ppy_noise)}")
    print(RMSE(real_x, real_y, X_filtered,Y_filtered))
    print((points_MSE(real_x, real_y, X_filtered,Y_filtered)-points_MSE(real_x, real_y, geo_x,geo_y))/points_MSE(real_x, real_y, geo_x,geo_y))
    position_posterior_x_est = wavelet_noising(position_posterior_x_est,1.1)
    position_posterior_y_est = wavelet_noising(position_posterior_y_est, 1.1)
    print(f" dynamic filtering dynamic loactioning(MSE)："
          f"{points_MSE(real_x, real_y, position_posterior_x_est, position_posterior_y_est)},"
          f"snr：{signaltonoise(position_posterior_x_est,add_noise(position_posterior_x_est,0.2,0)), signaltonoise(position_posterior_y_est,add_noise(position_posterior_x_est,0.1,0))}")
    print((points_MSE(real_x, real_y, position_posterior_x_est, position_posterior_y_est)-points_MSE(real_x, real_y, geo_x,geo_y))/points_MSE(real_x, real_y, geo_x,geo_y))
    print(RMSE(real_x, real_y, position_posterior_x_est, position_posterior_y_est))
    plot(len(real_x))
    plt.show()


