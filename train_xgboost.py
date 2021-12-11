from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier
from matplotlib import pyplot
import xgboost as xgb
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


def train(dtrain, dtest, test_y):
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
    bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=100)
    # t2=time.time()
    ypred = bst.predict(dtest)

    y_pred = (ypred >= 0.5) * 1
    # t3=time.time()

    # tn, fp, fn, tp = metrics.confusion_matrix(test_y, y_pred).ravel()
    # print(str(tn) + ' ' + str(fp) + ' ' + str(fn) + ' ' + str(tp))
    # print('BAC: %.4f' % (0.5 * tp / (tp + fn) + 0.5 * tn / (tn + fp)))
    '''filehandle.write('ACC: %.4f\n' % metrics.accuracy_score(test_y, y_pred))
    filehandle.write('Recall: %.4f\n' % metrics.recall_score(test_y, y_pred))
    filehandle.write('F1-score: %.4f\n' % metrics.f1_score(test_y, y_pred))
    filehandle.write('Precesion: %.4f\n' % metrics.precision_score(test_y, y_pred))'''
    # for i in range(len(test_y)):
    #     print(str(test_y[i]) + ' ' + str(y_pred[i]))
        # filehandle.write(str(test_y[i]) + ' ' + str(y_pred[i]) + '\n')
    # filehandle.write('\n')
    fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    fp = 0
    for i in range(len(test_y)):
        if test_y[i] == 1 and y_pred[i] == 0:
            fp += 1
    # print('AUC: %.4f' % metrics.roc_auc_score(test_y, ypred))
    # print('ACC: %.4f' % metrics.accuracy_score(test_y, y_pred))  #
    # print('Recall: %.4f' % metrics.recall_score(test_y, y_pred))  #
    # print('F1-score: %.4f' % metrics.f1_score(test_y, y_pred))  #
    # print('Precesion: %.4f' % metrics.precision_score(test_y, y_pred))  #
    print('%.4f' % metrics.accuracy_score(test_y, y_pred))  #
    print('%.4f' % eer)
    print('%.4f' % fpr[1])
    print()
    # print(fp)
    # print(t2-t1)
    # print((t3-t2)/len(test_x))

    return metrics.accuracy_score(test_y, y_pred), metrics.recall_score(test_y, y_pred), \
           metrics.f1_score(test_y, y_pred), metrics.precision_score(test_y, y_pred), eer, fp


def modelfit(alg, train_x, train_y, test_x, test_y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_x, label=train_y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(train_x, train_y, eval_metric='auc')

    # Predict test set:
    dtrain_predictions = alg.predict(train_x)
    dtest_predictions = alg.predict(test_x)
    dtest_predprob = alg.predict_proba(test_x)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_y, dtrain_predictions))
    print("AUC Score : %f" % metrics.roc_auc_score(test_y, dtest_predprob))
    print("Accuracy (Test): %.4g" % metrics.accuracy_score(test_y, dtest_predictions))
    print('Recall: %.4f' % metrics.recall_score(test_y, dtest_predictions))
    print('F1-score: %.4f' % metrics.f1_score(test_y, dtest_predictions))
    print('Precesion: %.4f' % metrics.precision_score(test_y, dtest_predictions))

    a = alg.feature_importances_
    for i in range(len(a)):
        if a[i] != 0:
            print(str(i))
            #print(str(i) + ' ' + str(a[i]))


def trainTestData(name, time, posture):
    file = filepath + name + '.txt'
    fs = open(file, 'r')
    str_data = fs.read()
    data = str_data.split('\n')
    train_x, train_y, test_x, test_y = [], [], [], []

    for item in data:
        temp = item.split()
        if not temp:
            break
        accelSquare = np.array([float(temp[i]) for i in range(2, len(temp))])
        if str(posture) == temp[1]:
            if time in temp[0]:
                train_x.append(accelSquare[high_freq_start:])
                train_y.append(1)#0
            elif '0516' in temp[0] or '0515' in temp[0]:
                test_x.append(accelSquare[high_freq_start:])
                test_y.append(1)#0
        else:
            if time in temp[0]:
                train_x.append(accelSquare[high_freq_start:])
                train_y.append(1)
            elif '0516' in temp[0] or '0515' in temp[0]:
                test_x.append(accelSquare[high_freq_start:])
                test_y.append(1)

    '''for item in namelist:
        if item != name:
            fs = open(filepath + item + '.txt', 'r')
            str_data = fs.read()
            dataOthers = str_data.split('\n')
            for dataItem in dataOthers:
                temp = dataItem.split()
                if not temp:
                    break
                if "0515" in temp[0] or "0516" in temp[0]:
                    accelSquare = np.array([float(temp[i]) for i in range(2, len(temp))])
                    train_y.append(1)
                    train_x.append(accelSquare)
                else:
                    continue'''
    return train_x, train_y
    #return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)


def trainTestHighFreqData(name, posture, data):
    # file = filepath + name + '.txt'
    # fs = open(file, 'r')

    train_x, train_y = [], []
    real = []

    for item in data:
        temp = item.split()
        if not temp:
            break

        accelSquare = np.array([float(temp[i]) for i in range(2, len(temp))])
        train_x.append(accelSquare[high_freq_start:])
        if str(posture) == temp[1] and name in temp[0]:
            train_y.append(0)
            real.append(0)
        else:
            train_y.append(1)

    return np.array(train_x), np.array(train_y), real


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

post = sys.argv[1]
# filepath = 'app/zhuyifeng/'
# print(filepath)
print(post)
postive_raw = readLog(post)
sample, l = len(postive_raw), len(postive_raw[0])
print('%d %d' % (sample, l))

subsize = 50
print('Size: %d' % subsize)
half = 60
for i in range(10):
    postive = random.sample(postive_raw[:half], k=subsize)
    print('ll: %d' % len(postive))

    negative = [[random.randrange(0, 100, 1)/100 for i in range(l)] for j in range(subsize)]
    print('%d %d' % (len(postive), len(negative)))

    training = np.array(postive + negative)
    label = np.array([1]*len(postive) + [0]*len(negative))
    dtrain = xgb.DMatrix(training, label=label)

    test_x_pos = postive_raw[half:] #eadLog('walking/30min/' + post)
    print('test size: %d' % len(test_x_pos))
    test_x_neg = [[random.randrange(0, 100, 1)/100 for i in range(l)] for j in range(len(test_x_pos))]
    test_x = test_x_pos + test_x_neg
    test_y = np.array([1]*len(test_x_pos) + [0]*len(test_x_neg))
    dtest = xgb.DMatrix(test_x)

    # train_x, test_x, train_y, test_y = train_test_split(training, label, random_state=0)
    # dtrain = xgb.DMatrix(train_x, label=train_y)
    # dtest = xgb.DMatrix(test_x)
    # print(train_x.shape)
    # print(test_x.shape)
    train(dtrain, dtest, test_y)
