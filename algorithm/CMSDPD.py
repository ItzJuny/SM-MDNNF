#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
    CMSDPD算法实现.
    ---
    Implementation of CMSDPD algorithm.
    Based on paper:
        Ren Y Z, Yang J, Wang J, Wang L. AMR steganalysis based on second-order difference of pitch delay[J]. 
        IEEE Transactions on Information Forensics and Security, 2017, 12(6): 1345–1357
        https://ieeexplore.ieee.org/abstract/document/7774981/
    ---
    Code by: wujunyan, E-mail: wjy9754@stu.hqu.edu.cn.
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
import argparse


def msdpd169(intpd, msdpd, total_subframes):  # LAG
    so = np.zeros(total_subframes)
    ms = np.zeros(13)
    zy = np.zeros((13, 13))
    for i in range(total_subframes - 2):
        so[i] = intpd[i + 2] - 2 * intpd[i + 1] + intpd[i]
    for i in range(total_subframes - 2):
        for j in range(-6, 7, 1):
            if so[i] == j:
                ms[j + 6] = ms[j + 6] + 1
    for i in range(total_subframes - 2):
        for l in range(-6, 7, 1):
            for k in range(-6, 7, 1):
                if so[i] == l:
                    if so[i + 1] == k:
                        zy[l + 6][k + 6] = zy[l + 6][k + 6] + 1
    k = 0
    for i in range(13):
        for j in range(13):
            msdpd[k] = float(zy[i][j]) / ms[i]
            if ms[i] == 0:
                msdpd[k] = 0
            k = k + 1


def extract_feature(data, length):
    total_subframes = int(length * 50 * 4)
    data = data.reshape(len(data), -1)
    fea_dim = 169
    feature = np.zeros((len(data), fea_dim))
    for index in tqdm(range(len(data))):
        msdpd169(data[index, :], feature[index, :], total_subframes)
    return feature


def shuffle(x, y):
    datasets = list(zip(x, y))
    np.random.shuffle(datasets)
    x, y = zip(*datasets)
    x, y = np.array(x), np.array(y)
    return x, y


def metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print('Accuracy={:.4f},fpr={:.4f},fnr={:.4f},tn={},fp={},fn={},tp={}'.format(accuracy, fpr, fnr, tn, fp, fn, tp))


def load_data(path):
    files = [os.path.join(path, i) for i in os.listdir(path)]
    data = np.asarray([np.loadtxt(i) for i in files if os.path.isfile(i)])
    return data


def split_train_test(stego_data, cover_data, splitnum):
    x_test = np.vstack((stego_data[:splitnum], cover_data[:splitnum]))
    x_train = np.vstack((stego_data[splitnum:], cover_data[splitnum:]))
    y_test = np.hstack((np.ones(int(len(x_test) / 2)), -np.ones(int(len(x_test) / 2))))
    y_train = np.hstack((np.ones(int(len(x_train) / 2)), -np.ones(int(len(x_train) / 2))))
    x_test, x_train = x_test[:, :, -8:-4], x_train[:, :, -8:-4]
    return x_test, x_train, y_test, y_train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CMSDPD')
    parser.add_argument("--length", help="sample length (s)", default=1, type=float)
    parser.add_argument("--split", help="split train/test dataset", default=0.25, type=float)
    parser.add_argument("--stego", help="/path/to/stegodata",
                        default="/home/wujunyan/data/amrnb_stego/huang/1s/10", type=str)
    parser.add_argument("--cover", help="/path/to/coverdata",
                        default="/home/wujunyan/data/amrnb_stego/huang/1s/0", type=str)
    parser.add_argument("--restego", help="/path/to/recompressed/stegodata",
                        default="/home/wujunyan/data/amrnb_stego/huang/1s/10/rec", type=str)
    parser.add_argument("--recover", help="/path/to/recompressed/coverdata/",
                        default="/home/wujunyan/data/amrnb_stego/huang/1s/0/rec", type=str)
    opt = parser.parse_args()
    print(opt)
    print("Loading Dataset")
    stego_data = load_data(opt.stego)
    cover_data = load_data(opt.cover)
    restego_data = load_data(opt.restego)
    recover_data = load_data(opt.recover)
    splitnum = int(len(stego_data) * opt.split)
    x_test, x_train, y_test, y_train = split_train_test(stego_data, cover_data, splitnum)
    x_test_rec, x_train_rec, y_test_rec, y_train_rec = split_train_test(restego_data, recover_data, splitnum)
    print("Extracting feature")
    x_train_MSDPD = extract_feature(x_train, opt.length)
    x_test_MSDPD = extract_feature(x_test, opt.length)
    x_train_rec_MSDPD = extract_feature(x_train_rec, opt.length)
    x_test_rec_MSDPD = extract_feature(x_test_rec, opt.length)
    x_train = x_train_MSDPD - x_train_rec_MSDPD
    x_test = x_test_MSDPD - x_test_rec_MSDPD
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    print("training")
    clf_rbf = svm.SVC(kernel='rbf', gamma="scale")
    clf_rbf.fit(x_train, y_train)
    y_pred = clf_rbf.predict(x_test)
    metrics(y_pred, y_test)
