import numpy as np
import scipy
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import pandas as pd
from particle.pf3 import pf
from scipy.fftpack import fft
from sklearn.metrics import mean_squared_error, r2_score
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose, Quaternion,PoseWithCovarianceStamped

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

class KF:
    def __init__(self, Q = 4e-4, R=0.25):
        # 定义超参数
        self.Q = Q  # 4e-4
        self.R = R  # 0.25
        # 定义尺寸函数
        # 定义迭代的初始参数
        self.X_bar = 0
        self.Xbar = 0
        self.K = 0
        self.P_ = 1
        self.P = None
        # print(Z)

    def KFfiltering(self,raw_data):
        if self.Xbar_pre == None:
            self.Xbar_pre = raw_data
            return raw_data
        # 时间更新
        self.P_ = self.P_ + self.Q
        # 状态更新
        self.K  = self.P_ / (self.P_ + self.R)
        self.Xbar = self.X_bar + self.K  * (raw_data - self.X_bar)
        self.P = (1 - self.K ) * self.P_
        return self.Xbar


class autoFiltering:
    def __init__(self):
        self.order = 1  # best 1 or 2
        self.fs = 5 #  采样频率
        cutoff = 1  # 3.667
        self.Meassure = 5e-5
        with open(r"cutoff.csv") as file:
            self.cutoff_fre = pd.read_csv(file)
            # TODO load best cutoff
        self.dataShape = np.shape(self.cutoff_fre)
        self.BBN = BBN()
        self.BBN.load()
        self.KF = KF()
        # TODO load model

    def FFT(self, data):
        """
                对输入信号进行FFT
                :param Fs:  采样频率
                :param data:待FFT的序列
                :return:
                """
        L = len(data)  # 信号长度
        N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
        result = np.abs(fft(x=data, n=int(N))) / L * 2  # N点FFT
        axisFreq = np.arange(int(N / 2)) * self.fs / N  # 频率坐标
        result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
        return axisFreq, result

    def find_peak(self,amp):
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
        swimming = [amp[0], amp[1], amp[2]]
        for i in range(l - 2):
            if swimming[0] < swimming[1] and swimming[1] > swimming[2]:
                peak.append(swimming[1])
                loc.append(i + 1)
            swimming = [swimming[1], swimming[2], amp[i + 2]]
        return peak, loc

    def cal_percentage(self, amp):
        '''
        用于计算不同峰值对应的FFT面积占比，挑选所需峰值，方便根据需要定制低通滤波的截止频率
        :param amp: 一维被滤波对象的FFT振幅
        :return:
        '''
        peak, loc = find_peak(amp)
        l = np.shape(amp)[0]
        Sum = 0
        for i in range(l):
            Sum += amp[i]
        percentage = []
        for l in loc:
            energe = 0
            for n in range(0, l, 1):
                energe += amp[n]

            percentage.append(round(energe / Sum * 100, 2))
        print(percentage)
        print(loc)
        return percentage, loc


    def butter_lowpass_filter(self, data, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        # print(normal_cutoff)
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y


    def autoFiltering(self,rawdata):
        kf_value = self.KF.KFfiltering(rawdata)
        state = {"x","y","field.angular_velocity.x","field.angular_velocity.y","field.angular_velocity.z",
                  "field.linear_acceleration.x", "field.linear_acceleration.y", "field.linear_acceleration.z"}
        #TODO fill state with data
        state = pd.DataFrame.from_dict(state)
        cutoff = self.BBN(state.values).values
        lowpass_value = self.butter_lowpass_filter(rawdata,cutoff,self.fs,self.order)

        print(rawdata,kf_value,lowpass_value)


def relatively_calm():
    # 生成模拟测量值
    # 目标与机器人不动
    X = None
    over = 0
    with open(r"../data/left/uwb.csv") as file:
        data = pd.read_csv(file)
        X = data["y"]
        over = X.shape[0]
    Meassure = 1e-4
    print("variation:",X.var()**2)
    with open(r"../data/left/uwb.csv") as file:
        data = pd.read_csv(file)
        X = data["y"]
        over = X.shape[0]

    fs = 5
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)  # 创建子图1
    ax1.scatter(X.index, X.values)
    plt.grid()
    # 绘制数据分布图

    ax2 = fig.add_subplot(2, 1, 2)  # 创建子图2
    X.hist(bins=50, alpha=0.5, ax=ax2)
    X.plot(kind='kde', secondary_y=True, ax=ax2)
    plt.grid()
    plt.show()

    x = np.arange(over)

    Z = X

    # print(Z.shape)
    # Y = 1.5*np.ones((over,1))
    Y = np.zeros((over,1))
    # X = butter_lowpass_filter(Z, 0.05, fs, order)
    print(f"均方误差(MSE)：{mean_squared_error(Y, X)}")

    kf = KF(Z, over, 1e-5, Meassure)
    # x_out, PF, z_out = pf(Y.tolist(), X.tolist(), 0, x_R=Meassure,N=300)
    axisFreq, result = FFT(fs, Z.values)
    # ax[1].plot(axisFreq, result)
    percentage, loc = cal_percentage(result)

    print("对应第四个波峰的频率是:", axisFreq[loc[3]],"占据了FFT图面积的",percentage[3],"%")
    best_cutoff = axisFreq[loc[3]]
    X = butter_lowpass_filter(Z, 0.31, fs, order)
    print(f"滤波后均方误差(MSE)：{mean_squared_error(Y, X)}")
    print(range(0,len(x),5))
    tick = []
    s =0.0
    for i in range(len(x)):
        s +=0.20
        tick.append(round(s,1))
    fig, ax = plt.subplots()
    ax.plot(x, Z, label='measure', color='orange')
    # ax.set_xticklabels(tick)
    ax.set_xlabel(" ")
    ax.set_ylabel("m")
    # ax.plot(x, Y, label='true',color='pink')
    ax.set_title("Original signal")
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(x, X, label='lowpass')
    ax.plot(x, Z, label='measure',color='orange')
    ax.set_title("Filted signal")
    # ax.set_xticklabels(tick)
    ax.set_xlabel(" ")
    ax.set_ylabel("m")
    # ax.plot(x, Y, label='true',color='pink')
   # ax.plot(x, kf, label='KF')
    # ax.plot(x, PF[1:], label='PF')
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(axisFreq, result)
    # ax.scatter(axisFreq[loc[3]], result[loc[3]])
    ax.set_title("FFT")
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB")
    ax.legend()
    # plt.title("Lowpass filter simulation (static)")



# Setting standard filter requirements.
order = 1 #best 1 or 2
fs = 5
cutoff = 1#3.667



Meassure = 5e-5
over = 100
relatively_calm()

plt.show()
