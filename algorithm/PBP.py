#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
    PBP算法实现.
    @论文: 基于基音延迟奇偶特性的自适应多速率语音隐写分析
    @作者: 刘小康(1), 田晖(1), 刘杰(1), 李想(1), 卢璥(2).
    @单位: (1)国立华侨大学计算机科学与技术学院, (2)华侨大学网络技术中心.
    代码版权归华侨大学厦门市数据安全与区块链技术重点实验室所有.
    ---
    Implementation of PBP algorithm.
    Based on paper:
        Steganalysis of Adaptive Multiple-Rate Speech Using Parity of Pitch-Delay Value[C]//
        International Conference on Security and Privacy in New Computing Environments. Springer, Cham, 2019: 282-297.
        https://link.springer.com/chapter/10.1007/978-3-030-21373-2_21
    The Authors are Xiaokang Liu(1), Hui Tian(1), Jie Liu(1), Xiang Li(1), Jing Lu(2).
    The Authors (1) are  all with the College of Computer Science and Technology,National Huaqiao University.
    The Author (2) are with the Network Technology Center,National Huaqiao University.
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


def eo14(intpd, eoff, total_subframes):
    i, j, k, o, e = (0, 0, 0, 0, 0)
    eo0 = np.zeros(16)
    eo1 = np.zeros(2)
    eo2 = np.zeros(4)
    eo3 = np.zeros(8)
    eo4 = np.zeros(16)
    for i in range(total_subframes - 2):
        if k == 4:
            k = 0
        if k % 2 == 0:
            if intpd[i] % 2 == 0:
                eo0[0] = eo0[0] + 1
                if intpd[i + 1] % 2 == 0:
                    eo0[1] = eo0[1] + 1
        elif k % 2 != 0:
            if intpd[i] % 2 != 0:
                eo0[2] = eo0[2] + 1
                if intpd[i + 1] % 2 != 0:
                    eo0[3] = eo0[3] + 1
        if k == 0:
            if intpd[i] % 2 == 0:
                eo0[4] = eo0[4] + 1
                if intpd[i + 2] % 2 == 0:
                    eo0[5] = eo0[5] + 1
            elif intpd[i] % 2 != 0:
                eo0[6] = eo0[6] + 1
                if intpd[i + 2] % 2 != 0:
                    eo0[7] = eo0[7] + 1
        if k == 1:
            if intpd[i] % 2 == 0:
                eo0[8] = eo0[8] + 1
                if intpd[i + 2] % 2 == 0:
                    eo0[9] = eo0[9] + 1
            elif intpd[i] % 2 != 0:
                eo0[10] = eo0[10] + 1
                if intpd[i + 2] % 2 != 0:
                    eo0[11] = eo0[11] + 1
        if i % 4 == 0:
            if intpd[i] % 2 == 0:
                eo0[12] = eo0[12] + 1
                if i + 4 != total_subframes:
                    if intpd[i + 4] % 2 == 0:
                        eo0[13] = eo0[13] + 1
            elif intpd[i] % 2 != 0:
                eo0[14] = eo0[14] + 1
                if i + 4 != total_subframes:
                    if intpd[i + 4] % 2 != 0:
                        eo0[15] = eo0[15] + 1
        k = k + 1
    for i in range(total_subframes - 5):
        if intpd[i] % 2 == 0:
            eo1[0] = eo1[0] + 1
            if intpd[i + 1] % 2 == 0:
                eo2[0] = eo2[0] + 1
                if intpd[i + 2] % 2 == 0:
                    eo3[0] = eo3[0] + 1
                    if intpd[i + 3] % 2 == 0:
                        eo4[0] = eo4[0] + 1
                    elif intpd[i + 3] % 2 != 0:
                        eo4[1] = eo4[1] + 1
                elif intpd[i + 2] % 2 != 0:
                    eo3[1] = eo3[1] + 1
                    if intpd[i + 3] % 2 == 0:
                        eo4[2] = eo4[2] + 1
                    elif intpd[i + 3] % 2 != 0:
                        eo4[3] = eo4[3] + 1
            elif intpd[i + 1] % 2 != 0:
                eo2[1] = eo2[1] + 1
                if intpd[i + 2] % 2 == 0:
                    eo3[2] = eo3[2] + 1
                    if intpd[i + 3] % 2 == 0:
                        eo4[4] = eo4[4] + 1
                    elif intpd[i + 3] % 2 != 0:
                        eo4[5] = eo4[5] + 1
                elif intpd[i + 2] % 2 != 0:
                    eo3[3] = eo3[3] + 1
                    if intpd[i + 3] % 2 == 0:
                        eo4[6] = eo4[6] + 1
                    elif intpd[i + 3] % 2 != 0:
                        eo4[7] = eo4[7] + 1
        elif intpd[i] % 2 != 0:
            eo1[1] = eo1[1] + 1
            if intpd[i + 1] % 2 == 0:
                eo2[2] = eo2[2] + 1
                if intpd[i + 2] % 2 == 0:
                    eo3[4] = eo3[4] + 1
                    if intpd[i + 3] % 2 == 0:
                        eo4[8] = eo4[8] + 1
                    elif intpd[i + 3] % 2 != 0:
                        eo4[9] = eo4[9] + 1
                elif intpd[i + 2] % 2 != 0:
                    eo3[5] = eo3[5] + 1
                    if intpd[i + 3] % 2 == 0:
                        eo4[10] = eo4[10] + 1
                    elif intpd[i + 3] % 2 != 0:
                        eo4[11] = eo4[11] + 1
            elif intpd[i + 1] % 2 != 0:
                eo2[3] = eo2[3] + 1
                if intpd[i + 2] % 2 == 0:
                    eo3[6] = eo3[6] + 1
                    if intpd[i + 3] % 2 == 0:
                        eo4[12] = eo4[12] + 1
                    elif intpd[i + 3] % 2 != 0:
                        eo4[13] = eo4[13] + 1
                elif intpd[i + 2] % 2 != 0:
                    eo3[7] = eo3[7] + 1
                    if intpd[i + 3] % 2 == 0:
                        eo4[14] = eo4[14] + 1
                    elif intpd[i + 3] % 2 != 0:
                        eo4[15] = eo4[15] + 1
    for i in range(2):
        eoff[i] = eo2[i * 2] / eo1[i]
        if eo1[i] == 0:
            eoff[i] = 0
    j = 0
    for i in range(2, 6):
        eoff[i] = eo3[j] / eo2[int(j / 2)]
        if eo2[int(j / 2)] == 0:
            eoff[i] = 0
        j = j + 2
    j = 0
    for i in range(6, 14):
        eoff[i] = eo4[j] / eo3[int(j / 2)]
        if eo3[int(j / 2)] == 0:
            eoff[i] = 0
        j = j + 2


def extract_feature(intpd, length):
    total_subframes = int(length * 50 * 4)
    intpd = intpd.reshape(len(intpd), -1)
    fea_dim = 14
    feature = np.zeros((len(intpd), fea_dim))
    for index in tqdm(range(len(intpd))):
        eo14(intpd[index, :], feature[index, :], total_subframes)
    return feature


def shuffle(x, y):
    datasets = list(zip(x, y))
    np.random.shuffle(datasets)
    x, y = zip(*datasets)
    x, y = np.array(x, dtype='float'), np.array(y, dtype=int)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PBP')
    parser.add_argument("--length", help="sample length (s)", default=1, type=float)
    parser.add_argument("--split", help="split train/test dataset", default=0.25, type=float)
    parser.add_argument("--stego", help="/path/to/stegodata",
                        default="/home/wujunyan/data/amrnb_stego/huang/1s/10", type=str)
    parser.add_argument("--cover", help="/path/to/coverdata",
                        default="/home/wujunyan/data/amrnb_stego/huang/1s/0", type=str)
    opt = parser.parse_args()
    print(opt)
    print("Loading Dataset")
    stego_data = load_data(opt.stego)
    cover_data = load_data(opt.cover)
    splitnum = int(len(stego_data) * opt.split)
    x_test = np.vstack((stego_data[:splitnum], cover_data[:splitnum]))
    x_train = np.vstack((stego_data[splitnum:], cover_data[splitnum:]))
    y_test = np.hstack((np.ones(int(len(x_test) / 2)), -np.ones(int(len(x_test) / 2))))
    y_train = np.hstack((np.ones(int(len(x_train) / 2)), -np.ones(int(len(x_train) / 2))))
    x_test, x_train = x_test[:, :, -8:-4], x_train[:, :, -8:-4]
    print("Extracting feature")
    x_train = extract_feature(x_train, opt.length)
    x_test = extract_feature(x_test, opt.length)
    x_test, y_test = shuffle(x_test, y_test)
    x_train, y_train = shuffle(x_train, y_train)
    print("Training")
    clf_rbf = svm.SVC(kernel='rbf', gamma="scale")
    clf_rbf.fit(x_train, y_train)
    y_pred = clf_rbf.predict(x_test)
    metrics(y_pred, y_test)
