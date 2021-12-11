import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.ensemble import IsolationForest
# from sklearn.cross_validation import train_test_split
from sklearn import metrics
# from sklearn import cross_validation
# from sklearn.grid_search import GridSearchCV
# from xgboost.sklearn import XGBClassifier
from matplotlib import pyplot
# import xgboost as xgb
import numpy as np
import os
import sys
from scipy.optimize import brentq
from scipy.interpolate import interp1d
# from process import *
import random
import time

filepath = "calm"
size = 4097
high_freq_start = int(7*size/50)
namelist = ['G1', 'G2', 'G3', 'G4', 'G5']


def trainingData(name, posture):
    file = filepath + name + '.txt'
    fs = open(file, 'r')
    str_data = fs.read()
    data = str_data.split('\n')
    training, label = [], []

    for item in data:
        temp = item.split()
        if not temp:
            break
        if str(posture) == temp[1] and name in temp[0]:
            label.append(0)
        else:
            label.append(1)
        accelSquare = np.array([float(temp[i]) for i in range(2, len(temp))])
        training.append(accelSquare)

    return np.array(training), np.array(label)


def train(dtrain, dtest, outliter):
    t1=time.time()
    clf = IsolationForest()
    clf.fit(dtrain)
    t2=time.time()
    y_pred_train = clf.predict(dtrain)

    t3 = time.time()
    y_pred_test = clf.predict(dtest)
    y_pred_outliter = clf.predict(outliter)
    t4 = time.time()

    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outlier = y_pred_outliter[y_pred_outliter == 1].size

    print(n_error_train, n_error_test, n_error_outlier)

    test_y = np.array([1]*len(dtest) + [-1]*len(outliter))
    y_pred = np.append(y_pred_test, y_pred_outliter)
    fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print(eer)

    return (n_error_train, n_error_test, n_error_outlier, eer, t2-t1, t4-t3)


    # y_pred = (ypred >= 0.5) * 1
    # t3=time.time()

    # fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred, pos_label=1)
    # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # fp = 0
    # for i in range(len(test_y)):
    #     if test_y[i] == 1 and y_pred[i] == 0:
    #         fp += 1
    # print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))
    # print('ACC: %.4f' % metrics.accuracy_score(test_y, y_pred))  #
    # print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))  #
    # print('F1-score: %.4f' % metrics.f1_score(test_y, y_pred))  #
    # print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))  #
    # print('%.4f' % metrics.accuracy_score(test_y, y_pred))  #
    # print('%.4f' % eer)
    # print('%.4f' % fpr[1])
    # print()
    # print(fp)
    # print(t2-t1)
    # print((t3-t2)/len(test_x))

    # return metrics.accuracy_score(test_y, y_pred), metrics.recall_score(test_y, y_pred), \
    #        metrics.f1_score(test_y, y_pred), metrics.precision_score(test_y, y_pred), eer, fp


def readLog(filename):
    data = []
    with open("%s.txt" % filename, 'r') as fs:
        i = -1
        lines = fs.readlines()
        for item in lines:
            if not item:
                continue
            i += 1
            words = item.split()
            data.append([])
            for subitem in words:
                data[i].append(float(subitem))
        print('\t%d' % i)
    return data

post = 'SLR'
filepath = 'walking/0min/'
print(filepath)
print(post)
postive_raw = readLog(filepath+post)
sample, l = len(postive_raw), len(postive_raw[0])
print('%d %d' % (sample, l))

half = int(sample*3/4)
subsize = half-7
print('Size: %d' % subsize)
total_err_train, total_err_test_pos, total_err_test_neg = 0, 0, 0
total_train, total_test = 0, 0
total_eer = 0
total_t_train, total_t_test = 0, 0
for i in range(10):
    postive = random.sample(postive_raw[:half], k=subsize)
    training = np.array(postive)
    dtrain = training
    total_train += len(postive)
    print()
    print('ll: %d' % len(postive))


    test_x_pos = postive_raw[half:] #eadLog('walking/30min/' + post)
    test_x_neg = [[random.randrange(0, 100, 1)/100 for i in range(l)] for j in range(len(test_x_pos))]
    total_test += len(test_x_pos)*2
    print('test size: %d \t %d' % (len(test_x_pos), len(test_x_neg)))
    dtest = np.array(test_x_pos)
    outliter = np.array(test_x_neg)
    # test_x = test_x_pos + test_x_neg
    # test_y = np.array([1]*len(test_x_pos) + [0]*len(test_x_neg))
    # dtest = xgb.DMatrix(test_x)

    (err_train, err_test_pos, err_test_neg, eer, t_train, t_test) = train(dtrain, dtest, outliter)
    total_err_train += int(err_train)
    total_err_test_pos += int(err_test_pos)
    total_err_test_neg += int(err_test_neg)
    total_eer += eer
    total_t_train += t_train
    total_t_test += t_test

print('Training err: %.4f' % (total_err_train/total_train))
print('Test ACC: %.4f' % (1 - (total_err_test_pos+total_err_test_neg)/(total_test)))
print('ERR: %.4f' % (total_eer/10))
# print('Training time: %.4f' % (total_t_train/10*1000))
# print('Testing time: %.4f' % (total_t_test/10*1000))