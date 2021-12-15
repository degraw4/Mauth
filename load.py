import json
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy
from scipy import signal

# 7:  6+, 17, 20-
# 7: 5, 16, 
# 14: 4-, 5+, 12, 14-
# 20: 1-, 8, 12-, 16+
def load(name):
    # a3 b1 c3    
    # name = "a3"
    # f = open("../data/" + str(name) + ".json", encoding="utf-8")
    f = open("../data/" + str(name), encoding="utf-8")
    file = json.load(f)

    # 加速度计
    data_0_x = []
    data_0_y = []
    data_0_z = []

    # 磁力计
    data_1_x = []
    data_1_y = []
    data_1_z = []

    # 陀螺仪
    data_2_x = []
    data_2_y = []
    data_2_z = []

    for data in file[0]["collectSensorData"]:
        if data[0] == 0:
            data_0_x.append(data[1])
            data_0_y.append(data[2])
            data_0_z.append(data[3])
        elif data[0] == 1:
            data_1_x.append(data[1])
            data_1_y.append(data[2])
            data_1_z.append(data[3])
        else:
            data_2_x.append(data[1])
            data_2_y.append(data[2])
            data_2_z.append(data[3])
        
    acc = []
    for i in range(len(data_0_x)):
        a = math.sqrt(data_0_x[i] * data_0_x[i] + data_0_y[i] * data_0_y[i] + data_0_z[i] * data_0_z[i])
        acc.append(a)

    plt.figure()
    plt.plot(data_0_x, label = "acc x")
    plt.plot(data_0_y, label = "acc y")
    plt.plot(data_0_z, label = "acc z")
    #plt.plot(acc, label = "total acc")
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("samples")
    plt.ylabel("m2/s")
    plt.title("accelerometer")
    plt.savefig(str(name) + ".png")

    plt.figure()
    plt.plot(data_1_x, label = "mag x")
    plt.plot(data_1_y, label = "mag y")
    plt.plot(data_1_z, label = "mag z")
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("samples")
    plt.ylabel("uT")
    plt.title("magnetic")
    plt.savefig("mag.png")

    plt.figure()
    plt.plot(data_2_x, label = "gyro x")
    plt.plot(data_2_y, label = "gyro y")
    plt.plot(data_2_z, label = "gyro z")
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("samples")
    plt.ylabel("rad/s")
    plt.title("gyroscope")
    plt.savefig("gyro.png")

def fft(name, index):
    # a3 b1 c3    
    # name = "c3"
    # f = open("../data/" + str(name) + ".json", encoding="utf-8")
    f = open("../data/" + str(name), encoding="utf-8")
    file = json.load(f)

    # 加速度计
    data_0_x = []
    data_0_y = []
    data_0_z = []

    # 磁力计
    data_1_x = []
    data_1_y = []
    data_1_z = []

    # 陀螺仪
    data_2_x = []
    data_2_y = []
    data_2_z = []

    for data in file[0]["collectSensorData"]:
        if data[0] == 0:
            data_0_x.append(data[1])
            data_0_y.append(data[2])
            data_0_z.append(data[3])
        elif data[0] == 1:
            data_1_x.append(data[1])
            data_1_y.append(data[2])
            data_1_z.append(data[3])
        else:
            data_2_x.append(data[1])
            data_2_y.append(data[2])
            data_2_z.append(data[3])

    if index == "x":
        data = data_0_x
    elif index == "y":
        data = data_0_y
    elif index == "z":
        data = data_0_z
        
    N = len(data)
    T = 1.0 / 50.0
    
    y_f = scipy.fft.fft(data)
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)

    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]))
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("FFT")
    plt.savefig("fft.png")

    plt.figure()
    plt.plot(data, label = "acc")

    b, a = signal.butter(8, 0.4, 'highpass')   #配置滤波器 8表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data) 
    plt.plot(filtedData, label = "highpass")

    b, a = signal.butter(8, 0.4, 'lowpass')   #配置滤波器 8表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data) 
    plt.plot(filtedData, label = "highpass")
    
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("samples")
    plt.ylabel("m2/s")
    plt.title("accelerometer")
    plt.savefig("acc.png") 

    b, a = signal.butter(8, 0.4, 'highpass')   #配置滤波器 8表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data) 
    y_f = scipy.fft.fft(filtedData)
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)
    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]))
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("FFT highpass")
    plt.savefig("fft-h.png")


    b, a = signal.butter(8, 0.4, 'lowpass')   #配置滤波器 8表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data) 
    y_f = scipy.fft.fft(filtedData)
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)
    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]))
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("FFT lowpass")
    plt.savefig("fft-l.png")

def cfft_trans(data, window, overlap):
    num = 2 + (len(data) - window) // (window - overlap)
    cross = [[0 for i in range(len(data))] for i in range(num)]
    number = 0
    start = 0
    count = 0
    i = 0
    while i < len(data):
        cross[number][start] = data[i]
        count += 1
        start += 1
        i += 1
        if count == window:
            i -= 5
            number += 1
            start = i
            count = 0

    y_f = np.zeros((len(data)), dtype = complex)
    for i in range(len(cross)):
        y_f += scipy.fft.fft(cross[i])

    N = len(data)
    T = 1.0 / 50.0
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint = False)
    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]))
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("CFFT")
    plt.savefig("cfft.png")

def cfft():
    # a3 b1 c3    
    name = "c3"
    f = open("../data/" + str(name) + ".json", encoding="utf-8")
    file = json.load(f)

    # 加速度计
    data_0_x = []
    data_0_y = []
    data_0_z = []

    # 磁力计
    data_1_x = []
    data_1_y = []
    data_1_z = []

    # 陀螺仪
    data_2_x = []
    data_2_y = []
    data_2_z = []

    for data in file[0]["collectSensorData"]:
        if data[0] == 0:
            data_0_x.append(data[1])
            data_0_y.append(data[2])
            data_0_z.append(data[3])
        elif data[0] == 1:
            data_1_x.append(data[1])
            data_1_y.append(data[2])
            data_1_z.append(data[3])
        else:
            data_2_x.append(data[1])
            data_2_y.append(data[2])
            data_2_z.append(data[3])
    
    # 选择data
    data = data_0_y
    N = len(data)
    T = 1.0 / 50.0

    b, a = signal.butter(8, 0.4, 'highpass')   #配置滤波器 8表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)

    y_f = scipy.fft.fft(filtedData)
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)
    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]))
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("FFT highpass")
    plt.savefig("fft-h.png")

    cfft_trans(filtedData, 25, 5)

def cfft_trans_test(data, window, overlap):
    N = len(data)
    T = 1.0 / 50.0
    num = 2 + (len(data) - window) // (window - overlap)
    cross = [[0 for i in range(len(data))] for i in range(num)]
    number = 0
    start = 0
    count = 0
    i = 0
    while i < len(data):
        cross[number][start] = data[i]
        count += 1
        start += 1
        i += 1
        if count == window:
            i -= 5
            number += 1
            start = i
            count = 0

    y_f = scipy.fft.fft(data)
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)
    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]), label = "FFT")

    y_f = np.zeros((len(data)), dtype = complex)
    for i in range(len(cross)):
        y_f += scipy.fft.fft(cross[i])

    N = len(data)
    T = 1.0 / 50.0
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint = False)
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]), label = "CFFT")
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("FFT & CFFT")
    plt.savefig("cfft1.png")

def cfft_trans_test2(data, window, overlap):
    N = len(data)
    T = 1.0 / 50.0
    num = 2 + (len(data) - window) // (window - overlap)
    cross = [[0 for i in range(len(data))] for i in range(num)]
    number = 0
    start = 0
    count = 0
    i = 0
    while i < len(data):
        cross[number][start] = data[i]
        count += 1
        start += 1
        i += 1
        if count == window:
            i -= 5
            number += 1
            start = i
            count = 0

    y_f = scipy.fft.fft(data)
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)
    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]), label = "FFT")

    y_f = np.zeros((len(data)), dtype = complex)
    for i in range(len(cross)):
        y_f += scipy.fft.fft(cross[i])

    N = len(data)
    T = 1.0 / 50.0
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint = False)
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]), label = "CFFT")
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("FFT & CFFT")
    plt.savefig("cfft2.png")

# fft and cfft
def test(name, index):
    # # a3 b1 c3    
    # name = "c3"
    # f = open("../data/" + str(name) + ".json", encoding="utf-8")
    f = open("../data/" + str(name), encoding="utf-8")
    file = json.load(f)

    # 加速度计
    data_0_x = []
    data_0_y = []
    data_0_z = []

    # 磁力计
    data_1_x = []
    data_1_y = []
    data_1_z = []

    # 陀螺仪
    data_2_x = []
    data_2_y = []
    data_2_z = []

    for data in file[0]["collectSensorData"]:
        if data[0] == 0:
            data_0_x.append(data[1])
            data_0_y.append(data[2])
            data_0_z.append(data[3])
        elif data[0] == 1:
            data_1_x.append(data[1])
            data_1_y.append(data[2])
            data_1_z.append(data[3])
        else:
            data_2_x.append(data[1])
            data_2_y.append(data[2])
            data_2_z.append(data[3])
    
    if index == "x":
        data = data_0_x
    elif index == "y":
        data = data_0_y
    elif index == "z":
        data = data_0_z

    # 选择data
    b, a = signal.butter(8, 0.4, 'highpass')   #配置滤波器 8表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)
    N = len(data)
    T = 1.0 / 50.0

    cfft_trans_test(data, 40, 20)
    cfft_trans_test2(filtedData, 40, 20)

def velocity(name, index):
    # a3 b1 c3    
    # name = "a3"
    # f = open("../data/" + str(name) + ".json", encoding="utf-8")
    f = open("../data/" + str(name), encoding="utf-8")
    file = json.load(f)

    # 加速度计
    data_0_x = []
    data_0_y = []
    data_0_z = []

    # 磁力计
    data_1_x = []
    data_1_y = []
    data_1_z = []

    # 陀螺仪
    data_2_x = []
    data_2_y = []
    data_2_z = []

    for data in file[0]["collectSensorData"]:
        if data[0] == 0:
            data_0_x.append(data[1])
            data_0_y.append(data[2])
            data_0_z.append(data[3])
        elif data[0] == 1:
            data_1_x.append(data[1])
            data_1_y.append(data[2])
            data_1_z.append(data[3])
        else:
            data_2_x.append(data[1])
            data_2_y.append(data[2])
            data_2_z.append(data[3])
        
    if index == "x":
        data = data_0_x
    elif index == "y":
        data = data_0_y
    elif index == "z":
        data = data_0_z

    N = len(data)
    T = 1.0 / 50.0  # s

    # for i in range(len(data_2_z)):
    #     if i < len(data_2_z) - 1 and data_2_z[i] * data_2_z[i+1] <= 0:
    #         print(i)

    # for i in range(1, len(data_2_z)):
    #     if i < len(data_2_z) - 1 and data_2_z[i] >= data_2_z[i+1] and data_2_z[i] >= data_2_z[i-1]:
    #         print(i)

    v = [ 0 for i in range(N) ]
    v_cor = [ 0 for i in range(N) ]
    r = [ 0 for i in range(N) ]
    
    start = 74
    end = 132

    for i in range(start, N):
        v[i] = T * data[i] + v[i-1]

    delta = (v[end] - v[start]) / (end - start)
    
    for i in range(start, N):
        v_cor[i] = v[i] - (i - start) * delta

    for i in range(0, N):
        r[i] = math.fabs(v_cor[i]) / math.fabs(data_2_z[i])

    v_cor_smooth = scipy.signal.savgol_filter(v_cor, 9, 6)  
    w_smooth = scipy.signal.savgol_filter(data_2_z, 9, 6)  

    r_smooth = [ 0 for i in range(N) ]
    for i in range(0, N):
        r_smooth[i] = math.fabs(v_cor_smooth[i]) / math.fabs(w_smooth[i])

    plt.figure()
    plt.plot(v[:end], label = "velocity " + str(index))
    plt.plot(v_cor[:end], label = "correct velocity " + str(index))
    plt.plot(v_cor_smooth[:end], label = "smoothed correct velocity " + str(index))
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("samples")
    plt.ylabel("m/s")
    plt.title("velocity " + str(index))
    plt.savefig("v.png")

    plt.figure()
    plt.plot(data_2_z, label = "w")
    plt.plot(w_smooth, label = "w_smooth")
    # plt.plot(v_cor, label = "v")
    # plt.plot(data_2_z, label = "w")
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("samples")
    plt.ylabel("rad/s")
    plt.title("w")
    plt.savefig("w.png")

    plt.figure()
    plt.plot(r, label = "r")
    plt.plot(r_smooth, label = "r_smooth")
    # plt.plot(v_cor, label = "v")
    # plt.plot(data_2_z, label = "w")
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("samples")
    plt.ylabel("m")
    plt.title("r")
    plt.savefig("r.png")

    ave = 0
    for i in range(start, end):
        if r_smooth[i] < 2:
            ave += r_smooth[i]
    ave = ave / (end - start - 1)
    print(ave)

    plt.figure()
    plt.plot(r_smooth[start : end], label = "r_smooth")
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("samples")
    plt.ylabel("m")
    plt.title("r")
    plt.savefig("r_smooth.png")

    return v_cor

def radius(name, v1, v2):
    f = open("../data/" + str(name), encoding="utf-8")
    file = json.load(f)

    # 加速度计
    data_0_x = []
    data_0_y = []
    data_0_z = []

    # 磁力计
    data_1_x = []
    data_1_y = []
    data_1_z = []

    # 陀螺仪
    data_2_x = []
    data_2_y = []
    data_2_z = []

    for data in file[0]["collectSensorData"]:
        if data[0] == 0:
            data_0_x.append(data[1])
            data_0_y.append(data[2])
            data_0_z.append(data[3])
        elif data[0] == 1:
            data_1_x.append(data[1])
            data_1_y.append(data[2])
            data_1_z.append(data[3])
        else:
            data_2_x.append(data[1])
            data_2_y.append(data[2])
            data_2_z.append(data[3])

    N = len(data_0_x)
    T = 1.0 / 50.0  # s

    v = [ 0 for i in range(N)]
    r = [ 0 for i in range(N)]

    for i in range(N):
        v[i] = math.sqrt(v1[i] * v1[i] + v2[i] * v2[i])
        r[i] = v1[i] / data_2_z[i]

    plt.figure()
    plt.plot(r, label = "r")
    plt.plot(v1, label = "v")
    plt.plot(data_2_z, label = "w")
    plt.legend(loc = 0, ncol = 1)
    plt.xlabel("samples")
    plt.ylabel("m")
    plt.title("r ")
    plt.savefig("r.png")

if __name__ == '__main__':
    name = "14 (5).blob"
    # # load("7 (6).blob")
    # fft(name, "x")
    # test(name, "x")
    # load(name)
    v1 = velocity(name, "y")
    # v2 = velocity(name, "y")
    # radius(name, v1, v2)
