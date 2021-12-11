#python2.7
import sys
sys.path.append('./CombinedOneClass/oneclass')
import oneclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
# from sklearn.cross_validation import train_test_split
from sklearn import metrics
# from sklearn import cross_validation
# from sklearn.grid_search import GridSearchCV
# from xgboost.sklearn import XGBClassifier
from matplotlib import pyplot
# import xgboost as xgb
import numpy as np
import os
from scipy.optimize import brentq
from scipy.interpolate import interp1d
# from process import *
import random
import time

filepath = "calm"
size = 4097
high_freq_start = int(7*size/50)
namelist = ['G1', 'G2', 'G3', 'G4', 'G5']


# derive training data
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
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'min_child_weight': 1,
              'scale_pos_weight': 1,
              'eta': 0.1,
              'gamma': 0,
              'seed': 0,
              'nthread': 8,
              'alpha': 0,
              'silent': 1,
              'n_estimators': 1000}

    watchlist = [(dtrain, 'train')]

    # t1=time.time()
    clf = oneclass.OneClassClassifier(density_only=True)
    clf.fit(dtrain)
    # t2=time.time()
    y_pred_train = clf.predict(dtrain)
    y_pred_test = clf.predict(dtest)
    y_pred_outliter = clf.predict(outliter)

    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outlier = y_pred_outliter[y_pred_outliter == 1].size

    print n_error_train, n_error_test, n_error_outlier

    test_y = np.array([1]*len(dtest) + [-1]*len(outliter))
    y_pred = np.append(y_pred_test, y_pred_outliter)
    fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print eer

    return (n_error_train, n_error_test, n_error_outlier, eer)


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

# testing
post = sys.argv[1]
# filepath = 'app/zhuyifeng/'
# print(filepath)
print post
postive_raw = readLog(post)
sample, l = len(postive_raw), len(postive_raw[0])
print '%d %d' % (sample, l)

# calculate err/acc and other metrics 
subsize = 50
print 'Size: %d' % subsize
half = 60
total_err_train, total_err_test_pos, total_err_test_neg = 0, 0, 0
total_train, total_test = 0, 0
total_eer = 0
for i in range(10):
    postive = random.sample(postive_raw[:half], k=subsize)
    training = np.array(postive)
    dtrain = training
    total_train += len(postive)
    print
    print 'll: %d' % len(postive)


    test_x_pos = postive_raw[half:] #eadLog('walking/30min/' + post)
    test_x_neg = [[random.randrange(0, 100, 1)/100 for i in range(l)] for j in range(len(test_x_pos))]
    total_test += len(test_x_pos)*2
    print 'test size: %d \t %d' % (len(test_x_pos), len(test_x_neg))
    dtest = np.array(test_x_pos)
    outliter = np.array(test_x_neg)
    # test_x = test_x_pos + test_x_neg
    # test_y = np.array([1]*len(test_x_pos) + [0]*len(test_x_neg))
    # dtest = xgb.DMatrix(test_x)

    (err_train, err_test_pos, err_test_neg, eer) = train(dtrain, dtest, outliter)
    total_err_train += int(err_train)
    total_err_test_pos += int(err_test_pos)
    total_err_test_neg += int(err_test_neg)
    total_eer += eer

print 'Training err: %.4f' % (total_err_train/total_train)
print 'Test ACC: %.4f' % (1 - (total_err_test_pos+total_err_test_neg)/(total_test))
print 'ERR: %.4f' % (total_eer/10)