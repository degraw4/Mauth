import json
import math
import sys
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import scipy.fft
import scipy
from scipy import signal

win_size = 100  #512
# fft_size = 512  #8192
slide_win = 1#10

def load(name):
    # a3 b1 c3    
    # name = "a3"
    # f = open("../data/" + str(name) + ".json", encoding="utf-8")
    f = open(str(name), encoding="utf-8")
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

    for data in file:
        try:
            data_0_x.append(data["x"])
            data_0_y.append(data["y"])
            data_0_z.append(data["z"])
        except KeyError:
            pass
            
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

def mauth(name):
    f = open(str(name), encoding="utf-8")
    file = json.load(f)

    # 加速度计
    dataX = []
    dataY = []
    dataZ = []

    for data in file:
        try:
            dataX.append(float(data["x"]))
            dataY.append(float(data["y"]))
            dataZ.append(float(data["z"]))
        except KeyError:
            pass

    # b, a = signal.butter(8, 0.2, 'highpass')   #配置滤波器 8表示滤波器的阶数
    # dataX = signal.filtfilt(b, a, dataX)
    # dataY = signal.filtfilt(b, a, dataY)
    # dataZ = signal.filtfilt(b, a, dataZ)

    n = len(dataX)
    fft_size = n
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataZ = np.array(dataZ)
    zeros = np.array([0]*(fft_size-win_size))
    fft_result = np.array([0]*fft_size)
    win = signal.hann(fft_size)

    for i in range(0, n-win_size, slide_win):
        currentX, currentY, currentZ = dataX[i:i+win_size], dataY[i:i+win_size], dataZ[i:i+win_size]
        current_max_x, current_max_y, current_max_z = max(currentX), max(currentY), max(currentZ)
        current_min_x, current_min_y, current_min_z = min(currentX), min(currentY), min(currentZ)
        current_norm_x = np.array([(currentX[i]-current_min_x)/(current_max_x-current_min_x) for i in range(len(currentX))])
        current_norm_y = np.array([(currentY[i]-current_min_y)/(current_max_y-current_min_y) for i in range(len(currentY))])
        current_norm_z = np.array([(currentZ[i]-current_min_z)/(current_max_z-current_min_z) for i in range(len(currentZ))])

        avgX = sum(current_norm_x) / win_size
        avgY = sum(current_norm_y) / win_size
        avgZ = sum(current_norm_z) / win_size
        for j in range(win_size):
            current_norm_x[j] = current_norm_x[j] - avgX
            current_norm_y[j] = current_norm_y[j] - avgY
            current_norm_z[j] = current_norm_z[j] - avgZ
        tempX = np.append(current_norm_x, zeros)*win
        tempY = np.append(current_norm_y, zeros)*win
        tempZ = np.append(current_norm_z, zeros)*win

        temp_fft_x = abs(fft(tempX))
        temp_fft_y = abs(fft(tempY))
        temp_fft_z = abs(fft(tempZ))
        temp_fft = [math.sqrt(temp_fft_x[i]**2+temp_fft_y[i]**2+temp_fft_z[i]**2) for i in range(len(temp_fft_x))]
        fft_result = temp_fft + fft_result

    fft_result = fft_result/int((n-win_size)/slide_win)

    T = 1.0 / 50.0
    N = len(dataX)
    y_f = fft_result[0:int(fft_size/2+1)]
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)
    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]))
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("CFFT")
    plt.savefig("mauth.png")

def mauth_filter(name):
    
    f = open(str(name), encoding="utf-8")
    file = json.load(f)

    # 加速度计
    dataX = []
    dataY = []
    dataZ = []

    for data in file:
        try:
            dataX.append(float(data["x"]))
            dataY.append(float(data["y"]))
            dataZ.append(float(data["z"]))
        except KeyError:
            pass

    b, a = signal.butter(8, 0.2, 'highpass')   #配置滤波器 8表示滤波器的阶数
    dataX = signal.filtfilt(b, a, dataX)
    dataY = signal.filtfilt(b, a, dataY)
    dataZ = signal.filtfilt(b, a, dataZ)

    n = len(dataX)
    fft_size = n
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataZ = np.array(dataZ)
    zeros = np.array([0]*(fft_size-win_size))
    fft_result = np.array([0]*fft_size)
    win = signal.hann(fft_size)

    for i in range(0, n-win_size, slide_win):
        currentX, currentY, currentZ = dataX[i:i+win_size], dataY[i:i+win_size], dataZ[i:i+win_size]
        current_max_x, current_max_y, current_max_z = max(currentX), max(currentY), max(currentZ)
        current_min_x, current_min_y, current_min_z = min(currentX), min(currentY), min(currentZ)
        current_norm_x = np.array([(currentX[i]-current_min_x)/(current_max_x-current_min_x) for i in range(len(currentX))])
        current_norm_y = np.array([(currentY[i]-current_min_y)/(current_max_y-current_min_y) for i in range(len(currentY))])
        current_norm_z = np.array([(currentZ[i]-current_min_z)/(current_max_z-current_min_z) for i in range(len(currentZ))])

        avgX = sum(current_norm_x) / win_size
        avgY = sum(current_norm_y) / win_size
        avgZ = sum(current_norm_z) / win_size
        for j in range(win_size):
            current_norm_x[j] = current_norm_x[j] - avgX
            current_norm_y[j] = current_norm_y[j] - avgY
            current_norm_z[j] = current_norm_z[j] - avgZ
        tempX = np.append(current_norm_x, zeros)*win
        tempY = np.append(current_norm_y, zeros)*win
        tempZ = np.append(current_norm_z, zeros)*win

        temp_fft_x = abs(fft(tempX))
        temp_fft_y = abs(fft(tempY))
        temp_fft_z = abs(fft(tempZ))
        temp_fft = [math.sqrt(temp_fft_x[i]**2+temp_fft_y[i]**2+temp_fft_z[i]**2) for i in range(len(temp_fft_x))]
        fft_result = temp_fft + fft_result

    fft_result = fft_result/int((n-win_size)/slide_win)

    T = 1.0 / 50.0
    N = len(dataX)
    y_f = fft_result[0:int(fft_size/2+1)]
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)
    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]))
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("CFFT_highpass")
    plt.savefig("mauth_hp.png")

def myfft(name, index):
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
    print(N)
    print(len(y_f))
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)
    print(len(x_f))

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
    
def cfft(name):
    f = open(str(name), encoding="utf-8")
    file = json.load(f)

    # 加速度计
    data_0_x = []
    data_0_y = []
    data_0_z = []

    for data in file:
        try:
            data_0_x.append(float(data["x"]))
            data_0_y.append(float(data["y"]))
            data_0_z.append(float(data["z"]))
        except KeyError:
            pass
    
    # 选择data
    data = data_0_y
    N = len(data)
    T = 1.0 / 50.0

    b, a = signal.butter(8, 0.4, 'highpass')   #配置滤波器 8表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)
    filtedData = data

    y_f = scipy.fft.fft(filtedData)
    x_f = np.linspace(0.0, 1.0/(2.0*T), N//2, endpoint=False)
    plt.figure()
    plt.plot(x_f, 2.0/N * np.abs(y_f[:N//2]))
    plt.xlabel("frequence/Hz")
    plt.ylabel("amplitude")
    plt.title("FFT")
    plt.savefig("fft.png")

    #cfft_trans(filtedData, 20, 5)

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
    name = "layx.json"
    mauth(name)
    mauth_filter(name)