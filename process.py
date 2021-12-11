''' Pre-process the raw tremor data
'''
import numpy as np
from numpy.fft import fft
from scipy import signal
import math
from matplotlib import pyplot
# from sklearn import preprocessing
import time

win_size = 100  #512
fft_size = 512  #8192
Fs = 100
slide_win = 1#10
sensitivity = 1048
g = 9.8
size = 4097
high_freq_start = int(7*size/50)

accel_sensitivity = 32767/2
gyro_sensitivity = 32767/250
radis = math.pi/180

old_macro_threshold = 0.3050

def getWin():
    return win_size, slide_win

# read from raw data file and return the accel data as
# [[accel_x], [accel_y], [accel_z]]
def load(filename):
    print(filename)
    accel_x, accel_y, accel_z = [], [], []
    with open("%s.txt" % filename, 'r') as fr:
        i = 0
        lines = fr.readlines()
        for item in lines:
            words = item.split()
            if len(words) <= 1:
                continue
            i += 1
            if i % 2 != 1:
                continue
            accel_x.append(float(words[3]))
            accel_y.append(float(words[4]))
            accel_z.append(float(words[5]))
        print('\ti %d' % i)
        print('\tlen %d' % len(accel_x))
    return [accel_x, accel_y, accel_z]

# helper function for plotting
def getXaxis():
    x = np.linspace(0, Fs, fft_size)[0:int(fft_size / 2 + 1)]
    return x

# Preprocess the accel data through normalization and fft, 
# return the frequency field data
def fft_normsquare(dataX, dataY, dataZ):
    n = len(dataX)
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

    return fft_result[0:int(fft_size/2+1)]

# helper function for plotting
def plotTimeField(filename):
    [dataX, dataY, dataZ] = load(filename)

    pyplot.subplot(211)
    pyplot.plot(dataX, label='x')
    pyplot.plot(dataY, label='y')
    pyplot.plot(dataZ, label='z')
    pyplot.grid()
    pyplot.legend()

    pyplot.subplot(212)
    threeAxis = [math.sqrt(dataX[i]**2+dataY[i]**2+dataZ[2]**2) for i in range(len(dataX))]
    mean = np.mean(threeAxis)
    threeAxis = [(threeAxis[i] - mean) for i in range(len(threeAxis))]
    pyplot.plot(threeAxis)
    pyplot.grid()
    return threeAxis


# separate still from motion
def std(filename, j):
    [dataX, dataY, dataZ] = loadAccelData(filename)

    threeAxis = [math.sqrt(dataX[i]**2+dataY[i]**2+dataZ[2]**2) for i in range(len(dataX))]
    stdArray = [np.std(threeAxis[i:i+j]) for i in range(len(threeAxis)-j)]
    t1 = np.percentile(stdArray, 10)
    t2 = np.percentile(stdArray, 20)
    t3 = np.percentile(stdArray, 50)
    t4 = np.percentile(stdArray, 60)
    t5 = np.percentile(stdArray, 80)
    t6 = np.percentile(stdArray, 90)
    print(t1, t2, t3, t4, t5, t6)
    # win 10

    cover = []
    i = 0
    size = 30 # 30s * 100Hz = 3000 points
    for item in stdArray:
        if item <= 0.04723:
            cover.append(0)
        else:
            cover.append(item)
            #dataX[i], dataY[i], dataZ[i] = 0, 0, 0
        i += 1
    print(max(stdArray))
    pyplot.subplot(2, 1, 1)
    pyplot.plot(threeAxis)
    pyplot.grid(True)
    pyplot.subplot(2, 1, 2)
    pyplot.plot(stdArray)
    pyplot.axhline(y=t1, color='blue', linestyle='--')
    pyplot.axhline(y=t2, color='green', linestyle='--')
    pyplot.axhline(y=t3, color='cyan', linestyle='--')
    pyplot.axhline(y=t4, color='limegreen', linestyle='--')
    pyplot.axhline(y=t5, color='purple', linestyle='--')
    pyplot.axhline(y=t6, color='yellow', linestyle='--')
    pyplot.axhline(y=0.04723, color='red', linestyle='-')
    pyplot.plot(cover)
    pyplot.grid(True)
    pyplot.show()

    still, sub = [], []
    for i in range(len(stdArray)):
        if stdArray[i] > 0.04723:
            if sub:
                sub.append(i-1)
                still.append(sub)
                sub = []
            else:
                continue
        else:
            if sub and i == (len(stdArray) - 1):
                sub.append(i)
                still.append(sub)
            elif sub:
                continue
            else:
                sub.append(i)

    fft_num = []
    length = []
    for item in still:
        pyplot.plot(item[0], 0, 'o')
        pyplot.plot(item[1], 0, 'x')
        #pyplot.show()
        temp = item[1]-item[0]
        if temp == 0:
            continue
        else:
            data_size = int(temp/3000)
            print("datasize: ", data_size)
            if data_size >= 1:#temp > 300:
                for i in range(data_size):
                    length.append(3000)
                    fft_num.append([item[0]+i*3000, item[0]+(i+1)*3000])
                    #length.append(item[1] - item[0])
                    #fft_num.append([item[0], item[1]])
                #j = int(temp/500)
                #for i in range(j):
                    #fft_num.append([item[0]+500*i, item[0]+500*(i+1)])
    print(len(length))
    #print(sum(length)/len(length))

    result = []
    i = 0
    for item in fft_num:
        result.append(fft_normsquare(dataX[item[0]:(item[1]+1)], dataY[item[0]:(item[1]+1)], dataZ[item[0]:(item[1]+1)])
                      [high_freq_start:])
        i += 1

    return result


def loadAccelData(filename):
    accel_x, accel_y, accel_z = [], [], []
    fr = open('%s.txt' % filename)
    i = 0
    for line in fr.readlines():
        i += 1
        line_arr = line.strip().split()
        if len(line_arr) <= 1:
            continue
        accel_x.append(float(line_arr[2]))
        accel_y.append(float(line_arr[5]))
        accel_z.append(float(line_arr[8]))
    fr.close()
    return [accel_x, accel_y, accel_z]


# name = ['pengmingran', 'wangxiao', 'xuchaoqi', 'zhuyifeng']

# time = [['0514', '0515', '0516', '0517'], ['0514', '0515', '0516', '0517'], 
#         ['0514', '0515', '0516', '0517', '0518', '0519'], 
#         ['0514', '0515', '0516', '0517']]
# num = [[3, 3, 3, 3], [3, 3, 3, 1], 
#         [3, 4, 3, 1, 1, 2], 
#         [5, 4, 3, 2]]
# post = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']
# size = 20*100
# window = 5*100
# for k in range(len(name)):
#     filepath = 'history/'+name[k]+'/'
#     for t in range(len(time[k])):
#         for p in range(len(post)):
#             print()
#             print(filepath)
#             print(time[k][t] + ' ' + post[p])
#             f = open((filepath+'%s_%s.txt') % (time[k][t], post[p]), 'w')
#             for i in range(num[k][t]):
#                 x, y, z = loadAccelData(filepath+time[k][t]+'_'+post[p]+'_'+str(i+1))
#                 print(len(x))
#                 for j in range(int((len(x)-window*2-size)/window)):
#                     print('\t%d' % i)
#                     temp_x = x[(i+1)*window:(i+1)*window+size]
#                     temp_y = y[(i+1)*window:(i+1)*window+size]
#                     temp_z = z[(i+1)*window:(i+1)*window+size]
#                     result = fft_normsquare(temp_x, temp_y, temp_z)
#                     for item in result:
#                         f.write(str(item) + ' ')
#                     f.write('\n')
#             f.close()

# test code...
size = 5*100
window = 5*100
st, ed = [], []
x,y,z = load('calm/G1/calm_g1_SHR')
print(len(x))
for i in range(int((len(x)-window*2-size)/window)):
    print('\t%d' % i)
    temp_x = x[(i+1)*window:(i+1)*window+size]
    temp_y = y[(i+1)*window:(i+1)*window+size]
    temp_z = z[(i+1)*window:(i+1)*window+size]
    st.append(time.time())
    result = fft_normsquare(temp_x, temp_y, temp_z)
    ed.append(time.time())
latency = [(ed[i]-st[i]) for i in range(len(st))]
print(np.mean(latency))

'''name = ['caodi', 'cf', 'changshan', 'hanyibo', 'hongzi', 'huxinggang', 'pakistan', 'xuchaoqi', 'zhangke', 'zhuyifeng']
time = {'caodi':'0705', 'cf':'0705', 'changshan':'0705', 'hanyibo':'0621', 'hongzi':'0620', 'huxinggang':'0705', 'pakistan':'0705', 'xuchaoqi':'0704', 'zhangke':'0705', 'zhuyifeng':'0625'}
app = ['qq', 'news', 'zhihu', 'weibo', 'video']
num = ['1', '2', '3', '4']

for n in range(len(name)):
    for a in range(len(app)):
        filepath = 'app/' + name[n] + '/'
        print()
        print(filepath)
        size = 20*100
        window = 15*100
        f = open((filepath+"%s.txt") % app[a], 'w')
        for i in num:
            x, y, z = load(filepath + time[name[n]]+'_'+app[a]+'_'+i)
            print(len(x))
            for i in range(int((len(x)-window*2-size)/window)):
                print('\t%d' % i)
                temp_x = x[(i+1)*window:(i+1)*window+size]
                temp_y = y[(i+1)*window:(i+1)*window+size]
                temp_z = z[(i+1)*window:(i+1)*window+size]
                result = fft_normsquare(temp_x, temp_y, temp_z)
                for item in result:
                    f.write(str(item) + ' ')
                f.write('\n')
        f.close()'''

'''motion-data
time = ['0', '30']
motion = ['pushup/', 'walking/']
inner = ['pushup', 'walk']
num = ['G1', 'G2', 'G3', 'G4', 'G5']
num2 = ['g1', 'g2', 'g3', 'g4', 'g5']
post = ['SHR', 'SLL', 'SLR', 'THR', 'TLL', 'TLR']
current_list = [0,1,2,3,4,5]

for m in range(2):
    for t in range(2):
        filepath = motion[m] + time[t] + "min/"
        print()
        print(filepath)
        for current in current_list:
filepath = 'pushup/0min/'
current = 0
size = 20*100
window = 5*100
f = open((filepath + "%s.txt") % post[current], 'w')
for i in range(len(num)):
    x, y, z = load(filepath + 'pushup_0m_' + post[current])
    print(len(x))
    for i in range(int((len(x)-window*2-size)/window)):
        print('\t%d' % i)
        temp_x = x[(i+1)*window:(i+1)*window+size]
        temp_y = y[(i+1)*window:(i+1)*window+size]
        temp_z = z[(i+1)*window:(i+1)*window+size]
        result = fft_normsquare(temp_x, temp_y, temp_z)
        for item in result:
            f.write(str(item) + ' ')
        f.write('\n')
f.close()'''

'''length = []
pyplot.figure('1')
a = plotTimeField('macro.txt')
length = std('macro.txt', 10)
f = open('macro.txt', 'a')
line = ''
print(len(a))
print(len(length))
for item in a:
    line = line + str(item) + '\t'
line = line + '\n'
for item in length:
    line = line + str(item) + '\t'
line = line + '0\t'*10 + '\n'
f.write(line)
f.close()
pyplot.figure('2')
plotTimeField('app/hongzi/0620_211647_hongzi_qq_2.txt')
length += std('app/hongzi/0620_211647_hongzi_qq_2.txt', 10)
pyplot.figure('3')
plotTimeField('app/hongzi/0620_212238_hongzi_qq_3.txt')
length += std('app/hongzi/0620_212238_hongzi_qq_3.txt', 10)
pyplot.figure('4')
plotTimeField('app/hongzi/0620_212803_hongzi_qq_4.txt')
length += std('app/hongzi/0620_212803_hongzi_qq_4.txt', 10)

pyplot.figure('cdf_all')
pyplot.hist(length, normed=True, cumulative=True, histtype='step', bins=1000)
pyplot.grid()
pyplot.show()'''
