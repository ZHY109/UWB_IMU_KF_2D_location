#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   make_filtering.py    
@Contact :   1094404954@qq.com
@License :   (C)Copyright Cheng Hoiyuen

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/5/2 16:55   郑海远      1.0         None
'''
from copy import copy

import gif
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import pywt
import seaborn as sn
#sgn函数
from sklearn.svm import SVC

pd.options.display.notebook_repr_html=False  # 表格显示
plt.rcParams['figure.dpi'] = 130  # 图形分辨率
sn.set_theme(style='white')  # 图形主题

def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising(new_df,threshold):
    data = new_df
    # data = data.values.T.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('sym6')
    maxlevel = pywt.dwt_max_level(len(data), w.dec_len)
    signal = pywt.wavedec(data, w, level=maxlevel)

    abs_cd1 = np.abs(np.array(signal[-1]))
    median_cd1 = np.median(abs_cd1)

    usecoeffs = []
    usecoeffs.append(signal[0])

    for i in range(1, len(signal)):
        usecoeffs.append(pywt.threshold(signal[i], threshold * max(signal[i])) ) # 将噪声滤波

    recoeffs = pywt.waverec(usecoeffs, w)#信号重构
    return recoeffs


def make_predict(df,inputs,name,modelname):
    if modelname == "xgb":
        model = xgb.XGBRegressor()
        model.load_model("../models/XGB/"+name)
        return model.predict(df[inputs])
    elif modelname == "right_svm":
        with open(r"../data/all.csv") as file:
            dataset = pd.read_csv(file)
            inputs = ["field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z",
                      "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z"]
            outputs = ["right_uwb_thread"]  # "left_uwb_thread", "right_uwb_thread"

        batch = 50
        x = dataset[inputs].values
        y = np.ravel(dataset[outputs].values * 10)
        model = SVC(C=100000, coef0= 0.0, degree=5, kernel='poly')
        model.fit(x,y)
        return model.predict(df[inputs].values) / 10
    elif modelname == "left_svm":
        with open(r"../data/all.csv") as file:
            dataset = pd.read_csv(file)
            inputs = ["field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z",
                      "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z"]
            outputs = ["left_uwb_thread"]  # "left_uwb_thread", "right_uwb_thread"

        batch = 50
        x = dataset[inputs].values
        y = np.ravel(dataset[outputs].values * 10)
        model = SVC(C=50000,coef0=0.0,degree=4,kernel='poly')
        model.fit(x, y)
        return model.predict(df[inputs].values) / 10

@gif.frame
def plot(i):
    imu_string = "IMU: " + str(dataset[inputs].values[i])
    fig, axs = plt.subplots(figsize=(6, 6))
    j = copy(i)
    x = np.arange(i)
    axs.plot(x, data2['r'][:i], "-", label="original UWB",linewidth=1)
    axs.plot(x, corrected_r[:i], "-", label="filtered",color='orange', linewidth=2)
    # plt.text(0, 0.5, imu_string,fontsize=9)
    plt.text(0, 0.5, "cutoff: "+str(filter_para_r[i]),color='g', fontsize=20)
    if i>window_size:
        axs.plot(x[i - window_size:i], data2['r'][i - window_size:i], ".",label="sliding window", color='red', linewidth=1)

    axs.set_xlabel('No.')  # Add an x-label to the axes.
    # axs.set_ylabel('m')  # Add an x-label to the axes.
    axs.set_title("results")

    axs.legend()  # Add a legend.
    # plt.show()

with open(r"../data/2023_4_26/uwb.csv") as file:
    data2 = pd.read_csv(file)
    data2 = data2.dropna()
    print(data2.shape)

with open(r"../data/2023_4_26/corrected_uwb.csv") as file:
    data = pd.read_csv(file)
    inputs = ["field.angular_velocity.x", "field.angular_velocity.y", "field.angular_velocity.z",
              "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z"]
    name = ["left_uwb_thread","right_uwb_thread"]
    with open(r"../data/all.csv") as file:
        dataset = pd.read_csv(file)
    filter_para_l = make_predict(data,inputs,name[0],"left_svm")
    filter_para_r = make_predict(data, inputs, name[1],"right_svm")
    print(filter_para_r)
    corrected_l = []
    corrected_r = []
    r = []
    l = []
    olr = []
    oll = []
    length = data2['l'].shape[0]
    window_size = 5
    i=0
    while True:
        if i >= length:
            break
        if len(r)<window_size:
            r.append(data2['r'][:i].values)
            l.append(data2['l'][:i].values)
            i += 1
            continue
        if  i>length-window_size:
            r.append(data2['r'][i:].values)
            l.append(data2['l'][i:].values)
            i += 1
            continue
        r.append(data2['r'][i-window_size:i].values)
        l.append(data2['l'][i-window_size:i].values)
        i += 1
    print(len(l))
    olr = r.copy()
    oll = l.copy()
    frames = []
    for i in range(len(filter_para_l)):
        try:
            corrected_l.append(np.mean(wavelet_noising(l[i], filter_para_l[i])))
            corrected_r.append(np.mean(wavelet_noising(r[i], filter_para_r[i])))
        except:
            print(i)
            print(i)
        print(len(corrected_l[:i]),len(l[:i]))
        frame = plot(i)
        frames.append(frame)
        print(len(frames))
        # 根据帧序列frames,动画持续时间duration，生成gif动画
    gif.save(frames, 'make_filtering.gif', duration=10)

    # pd.DataFrame({'l': corrected_l, 'r': corrected_r,'filter_para_l':filter_para_l,'filter_para_r':filter_para_r, 'oll': oll, 'olr': olr}).to_csv(
    #     "../data/2023_4_26/corrected222.csv")
    print("OK")
