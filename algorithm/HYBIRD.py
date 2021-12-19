#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
    HYBIRD算法实现.
    @论文: 基于基音延迟统计特性的自适应多速率语音隐写分析
    @作者: 田晖(1), 黄美伦(1), 刘杰(1), 张真诚(2), 黄永峰(3), 卢璥(4), 杜勇前(4)
    @单位: (1)国立华侨大学计算机科学与技术学院, (2)逢甲大学信息与计算机科学系, (3)清华大学电气工程系 , (4)华侨大学网络技术中心.
    代码版权归华侨大学厦门市数据安全与区块链技术重点实验室所有.
    ---
    Implementation of HYBIRD algorithm.
    Based on paper:
        Steganalysis of Adaptive Multi-Rate Speech Using Statistical Characteristics of Pitch Delay[J].
        Journal of Universal Computer Science, vol. 25, no. 9 (2019),1131-1150.
        https://lib.jucs.org/article/22649/download/pdf/
    The Authors are Hui Tian(1), Meilun Huang(1), Chin-Chen Chang(2), Yongfeng Huang(3), Jing Lu(4), Yongqian Du(4).
    The Authors (1) are  all with the College of Computer Science and Technology,National Huaqiao University.
    The Author (2) are with Department of lnformation and Computer Science, Feng Chia University.
    The Author (3) are  with Department of Electrical Engineering, Tsinghua University.
    The Authors (4) are all with the Network Technology Center, National Huaqiao University.
    Copyright 2021-2025 © Xiamen Key Laboratory of Data Security and Blockchain Technology, Huaqiao University
    All Rights Reserved.
    ---
    Code by: wujunyan, E-mail: wjy9754@stu.hqu.edu.cn.
"""


import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
import argparse


def msdpd9(intpd, msdpd, total_subframes):
    so = np.zeros(total_subframes)
    ms = np.zeros(3)
    zy = np.zeros((3, 3))
    for i in range(total_subframes - 2):
        so[i] = intpd[i + 2] - 2 * intpd[i + 1] + intpd[i]
    for i in range(total_subframes - 2):
        for j in range(-1, 2, 1):
            if so[i] == j:
                ms[j + 1] = ms[j + 1] + 1
    for i in range(total_subframes - 2):
        for l in range(-1, 2, 1):
            for k in range(-1, 2, 1):
                if so[i] == l:
                    if so[i + 1] == k:
                        zy[l + 1][k + 1] = zy[l + 1][k + 1] + 1
    k = 0
    for i in range(3):
        for j in range(3):
            msdpd[k] = float(zy[i][j]) / ms[i]
            if ms[i] == 0:
                msdpd[k] = 0
            k = k + 1


def extract_feature_msdpd(data, length):
    total_subframes = int(length * 50 * 4)
    data = data.reshape(len(data), -1)
    fea_dim = 9
    feature = np.zeros((len(data), fea_dim))
    for index in tqdm(range(len(data))):
        msdpd9(data[index, :], feature[index, :], total_subframes)
    return feature


def PDOEPD(intpd, pdoepd, total_subframes):
    frames = int(total_subframes / 4)
    intpd = intpd.reshape(frames, 4)
    v = np.zeros(frames)
    for i in range(frames):
        for j in range(4):
            if intpd[i, j] % 2 == 1:
                v[i] = v[i] + 1
    for i in range(5):
        pdoepd[i] = float(np.sum(v == i) / frames)


def extract_feature_pdoepd(data, length):
    total_subframes = int(length * 50 * 4)
    data = data.reshape(len(data), -1)
    fea_dim = 5
    feature = np.zeros((len(data), fea_dim))
    for index in tqdm(range(len(data))):
        PDOEPD(data[index, :], feature[index, :], total_subframes)
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
    parser = argparse.ArgumentParser(prog='HYBIRD')
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
    x_train_MSDPD = extract_feature_msdpd(x_train, opt.length)
    x_test_MSDPD = extract_feature_msdpd(x_test, opt.length)
    x_train_rec_MSDPD = extract_feature_msdpd(x_train_rec, opt.length)
    x_test_rec_MSDPD = extract_feature_msdpd(x_test_rec, opt.length)
    x_train_PDOEPD = extract_feature_pdoepd(x_train, opt.length)
    x_test_PDOEPD = extract_feature_pdoepd(x_test, opt.length)
    x_train, x_test = np.zeros((len(y_train), 9 + 5)), np.zeros((len(y_test), 9 + 5))
    x_train[:, :9], x_train[:, 9:] = x_train_MSDPD - x_train_rec_MSDPD, x_train_PDOEPD
    x_test[:, :9], x_test[:, 9:] = x_test_MSDPD - x_test_rec_MSDPD, x_test_PDOEPD
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    print("training")
    clf_rbf = svm.SVC(kernel='rbf', gamma="scale")
    clf_rbf.fit(x_train, y_train)
    y_pred = clf_rbf.predict(x_test)
    metrics(y_pred, y_test)
