import math

import numpy as np
import scipy
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import pandas as pd
from particle.pf3 import pf
from scipy.fftpack import fft
from sklearn.metrics import mean_squared_error, r2_score
import pywt
import seaborn as sn
sn.set_theme(style='darkgrid')  # 图形主题

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

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # print(normal_cutoff)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def KF(Z, over, Q, R):
    # 定义超参数
    over = over
    Q = Q  # 4e-4
    R = R  # 0.25
    # 定义尺寸函数
    cc = [over, 1]
    # 定义迭代的初始参数
    X_bar = np.zeros(cc)
    Xbar = np.zeros(cc)
    K = np.zeros(cc)
    P_ = np.zeros(cc)
    P = np.zeros(cc)
    P[0] = 1
    # print(Z)
    Xbar[0] = Z[0]

    # 循环迭代
    for n in range(1, over):
        # 时间更新
        X_bar[n] = Xbar[n - 1]
        P_[n] = P[n - 1] + Q
        # 状态更新
        K[n] = P_[n] / (P_[n] + R)
        Xbar[n] = X_bar[n] + K[n] * (Z[n] - X_bar[n])
        P[n] = (1 - K[n]) * P_[n]
    return Xbar

def plot(true,est,name,measure):
    len = np.shape(est)[0]
    x = np.arange(len)
    fig, ax = plt.subplots(len, 1)
    for i in range(len):
        ax[i].plot(x, est[i], label=name[i])
        ax[i].plot(x, measure[i], label='measure')
        ax[i].plot(x, true, label='true')
        ax[i].set_title(name[i])
        ax[i].legend()
    plt.show()

def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    L = len(data)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(fft(x=data, n=int(N))) / L * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    return axisFreq, result

def find_peak(amp):
    '''
    找到所有峰值
    :param amp:
    :return:
    '''
    peak = []
    peak.append(amp[0])
    loc = []
    loc.append(0)
    l = np.shape(amp)[0]
    swimming = [amp[0],amp[1],amp[2]]
    for i in range(l-2):
        if swimming[0]<swimming[1] and swimming[1]>swimming[2]:
            peak.append(swimming[1])
            loc.append(i+1)
        swimming = [swimming[1],swimming[2],amp[i+2]]
    return peak,loc

def cal_percentage(amp):
    '''
    用于计算不同峰值对应的FFT面积占比，挑选所需峰值，方便根据需要定制低通滤波的截止频率
    :param amp: 一维被滤波对象的FFT振幅
    :return:
    '''
    peak, loc = find_peak(amp)
    l = np.shape(amp)[0]
    Sum = 0
    for i in range(l):
        Sum+=amp[i]
    percentage = []
    for l in loc:
        energe = 0
        for n in range(0,l,1):
            energe+=amp[n]

        percentage.append(round(energe/Sum*100,2))
    print(percentage)
    print(loc)
    return percentage, loc

def relatively_calm():
    # 生成模拟测量值
    # 目标与机器人不动
    X = None
    over = 0
    with open(r"../data/right/uwb.csv") as file:
        data = pd.read_csv(file)
        X = data["right_length"]
        over = X.shape[0]
    Meassure = 1e-4
    print("variation:",X.var()**2)
    with open(r"../data/right/uwb.csv") as file:
        data = pd.read_csv(file)
        X = data["right_length"]
        over = X.shape[0]

    # print(X.shape)
    # fs = 5
    # fig = plt.figure(figsize=(10, 6))
    # ax1 = fig.add_subplot(2, 1, 1)  # 创建子图1
    # ax1.scatter(X.index, X.values)
    # plt.grid()
    # # 绘制数据分布图
    #
    # ax2 = fig.add_subplot(2, 1, 2)  # 创建子图2
    # X.hist(bins=50, alpha=0.5, ax=ax2)
    # X.plot(kind='kde', secondary_y=True, ax=ax2)
    # plt.grid()
    # plt.show()

    x = np.arange(over)

    Z = X

    # print(Z.shape)
    # Y = 1.5*np.ones((over,1))
    Y = np.ones((over,1))*0
    print(f"均方误差(MSE)：{mean_squared_error(Y, Z)}")

    kf = KF(Z, over, 1e-5, Meassure)


    # x_out, PF, z_out = pf(Y.tolist(), X.tolist(), 0, x_R=Meassure,N=300)
    axisFreq, result = FFT(fs, Z.values)
    # ax[1].plot(axisFreq, result)
    percentage, loc = cal_percentage(result)

    print("对应第四个波峰的频率是:", axisFreq[loc[3]],"占据了FFT图面积的",percentage[3],"%")
    best_cutoff = axisFreq[loc[3]]
    Xlowpass = butter_lowpass_filter(Z, 0.31, fs, order)

    print(f"lowpass滤波后均方误差(MSE)：{mean_squared_error(Y, Xlowpass)}")
    print(f"kf滤波后均方误差(MSE)：{mean_squared_error(Y, kf)}")
    X = pd.DataFrame(X)

    # tick = []
    # s =0.0
    # for i in range(len(x)):
    #     s +=0.20
    #     tick.append(round(s,1))
    #
    # data_denoising = wavelet_noising(Z, 1.1)
    # print(f"小波滤波后均方误差(MSE)：{mean_squared_error(Y, data_denoising)}")
    # m = min(data_denoising.shape[0], Y.shape[0])
    # print((mean_squared_error(Y, data_denoising)-mean_squared_error(Y, Z))/mean_squared_error(Y, Z))
    # import seaborn as sn
    # fig, ax = plt.subplots()
    # sn.plot(x, Z, label='measure', color='orange')
    # sn.plot(x[:m], data_denoising[:m], label='wavelet')
    # # ax.set_xticklabels(tick)
    #
    # sn.plot(x, kf, label='KF')
    # sn.plot(x, Xlowpass, label='lowpass')
    # sn.plot(x, Y, label='true',color='pink')
    # # ax.set_title("Original signal")
    # sn.set_xlabel(" ")
    # sn.set_ylabel("m")
    # sn.legend()
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(x, Z, label='measure', color='orange')
    # ax.plot(x[:m], data_denoising[:m], label='wavelet')
    # # ax.set_xticklabels(tick)
    # #
    # # ax.plot(x, kf, label='KF',color='blue')
    # # ax.plot(x, Xlowpass, label='lowpass')
    # ax.plot(x, Y, label='true',color='pink')
    # # ax.set_title("Original signal")
    # ax.set_xlabel(" ")
    # ax.set_ylabel("m")
    # ax.legend()

    fig, ax = plt.subplots()
    # ax.plot(x, X, label='lowpass')
    ax.plot(x, Z, label='measure',color='orange')
    ax.set_title("different wavelet cutoff")
    for i in range(2,12,1):
        data_denoising = wavelet_noising(Z, i/10)
        m = min(data_denoising.shape[0], Y.shape[0])
        ax.plot(x[:m], data_denoising[:m], label='wavelet'+str(i/10))

        print(f"wavelet滤波后均方误差(MSE)：{mean_squared_error(data_denoising[:m], Y[:m])}"+"\t"+str(i/10))
    ax.legend()
    # # 在[0.3,0.6]的小波阈值滤波效果最好 可以取0.6
    # ax.set_title("Filted signal")
    # # ax.set_xticklabels(tick)
    # ax.set_xlabel(" ")
    # ax.set_ylabel("m")
    # # ax.plot(x, Y, label='true',color='pink')
    # # ax.plot(x, kf, label='KF')
    # # ax.plot(x, PF[1:], label='PF')
    # ax.legend()
    #


# Setting standard filter requirements.
order = 1 #best 1 or 2
fs = 5
cutoff = 1#3.667



Meassure = 5e-5
over = 100
relatively_calm()

plt.show()
