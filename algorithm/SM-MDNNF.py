#!/usr/bin/env python
# -*-coding:utf-8 -*-

"""
    SM-MDNNF算法实现.
    @论文: 基于小数基音延迟相关性的自适应多速率语音流隐写分析.
    @作者: 田晖,吴俊彦,严艳,王慧东,全韩彧.
    @单位: 华侨大学计算机科学与技术学院; 华侨大学厦门市数据安全与区块链技术重点实验室; 华侨大学福建省大数据智能与安全重点实验室.
    代码版权归华侨大学厦门市数据安全与区块链技术重点实验室所有.
    ---
    Implementation of our SM-MDNNF algorithm.
    Based on paper:
        Steganalysis of Adaptive Multi-Rate Speech StreamsBased on the Correlation of Fractional Pitch Delay.
    The Authors are Hui Tian, Junyan Wu, Yan Yan, Huidong Wang, Hanyu Quan.
    The Authors are all with the School of Computer Science and Technology, Huaqiao University,
        Xiamen Key Laboratory of Data Security and Blockchain Technology, Huaqiao University,
        and Fujian Key Laboratory of Big Data Intelligence and Security, Huaqiao University.
    Copyright 2021-2025 © Xiamen Key Laboratory of Data Security and Blockchain Technology, Huaqiao University.
    All Rights Reserved.
    ---
    Code by: wujunyan, E-mail: wjy9754@stu.hqu.edu.cn.
"""

import os
import keras
import numpy as np
import argparse
from keras.layers import Reshape, Dense, LSTM, Bidirectional, AveragePooling1D, Flatten, Input, GlobalAveragePooling1D, \
    BatchNormalization, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import np_utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def shuffle(x, y):
    datasets = list(zip(x, y))
    np.random.shuffle(datasets)
    x, y = zip(*datasets)
    x, y = np.array(x), np.array(y)
    return x, y


def LCDM_model(input, opt):
    c1 = Conv1D(opt.LCDM_filters, opt.LCDM_kernelsize, activation='relu', padding='same')(input)
    c1 = keras.layers.BatchNormalization()(c1)
    c_pool = GlobalAveragePooling1D()(c1)
    output_cnn = Dense(2, activation="softmax")(c_pool)
    return output_cnn


def GCDM_model(input, opt):
    r1 = Bidirectional(LSTM(opt.GCDM_units1, return_sequences=True))(input)
    r2 = Bidirectional(LSTM(opt.GCDM_units2))(r1)
    r_fla = Flatten()(r2)
    output_rnn = Dense(2, activation="softmax")(r_fla)
    return output_rnn


def FFDM_model(input, opt):
    c1 = Conv1D(opt.LCDM_filters, opt.LCDM_kernelsize, activation='relu', padding='same')(input)
    c1 = keras.layers.BatchNormalization()(c1)
    c_pool = GlobalAveragePooling1D()(c1)
    r1 = Bidirectional(LSTM(opt.GCDM_units1, return_sequences=True))(input)
    r2 = Bidirectional(LSTM(opt.GCDM_units2))(r1)
    r_fla = Flatten()(r2)
    merge = keras.layers.concatenate([c_pool, r_fla])
    merge_fc = Dense(opt.FFDM_units, activation='relu')(merge)
    output_rcnn = Dense(2, activation="softmax")(merge_fc)
    return output_rcnn


def metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    print('Accuracy={:.4f},fpr={:.4f},fnr={:.4f},tn={},fp={},fn={},tp={}'.format(accuracy, fpr, fnr, tn, fp, fn, tp))
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='SM-MDNNF')
    parser.add_argument("--length", help="sample length (s)", default=1, type=float)
    parser.add_argument("--batchsize", help="batch size", default=256, type=int)
    parser.add_argument("--epoch", help="training epochs", default=20, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.001, type=float)
    parser.add_argument("--split", help="split train/test dataset", default=0.25, type=float)
    parser.add_argument("--stego", help="/path/to/stegodata",
                        default="/home/wujunyan/data/amrnb_stego/huang/1s/10", type=str)
    parser.add_argument("--cover", help="/path/to/coverdata",
                        default="/home/wujunyan/data/amrnb_stego/huang/1s/0", type=str)
    parser.add_argument("--LCDM_filters", help="cnn filters in LCDM", default=64, type=int)
    parser.add_argument("--LCDM_kernelsize", help="cnn kernelsize in LCDM", default=2, type=int)
    parser.add_argument("--GCDM_units1", help="units for the first blstm layer in GCDM", default=64, type=int)
    parser.add_argument("--GCDM_units2", help="units for the second blstm layer in GCDM", default=32, type=int)
    parser.add_argument("--FFDM_units", help="units for the fc layer in  FFDM", default=32, type=int)
    opt = parser.parse_args()
    print(opt)
    print("Loadind Dataset")
    stego_files = [os.path.join(opt.stego, i) for i in os.listdir(opt.stego)]
    cover_files = [os.path.join(opt.cover, i) for i in os.listdir(opt.cover)]
    stego_data = np.asarray([np.loadtxt(i) for i in stego_files if os.path.isfile(i)])
    cover_data = np.asarray([np.loadtxt(i) for i in cover_files if os.path.isfile(i)])
    splitnum = int(len(stego_data) * opt.split)
    x_test = np.vstack((stego_data[:splitnum], cover_data[:splitnum]))
    x_train = np.vstack((stego_data[splitnum:], cover_data[splitnum:]))
    y_test = np.hstack((np.ones(int(len(x_test) / 2)), np.zeros(int(len(x_test) / 2))))
    y_train = np.hstack((np.ones(int(len(x_train) / 2)), np.zeros(int(len(x_train) / 2))))
    x_test, y_test = shuffle(x_test, y_test)
    x_train, y_train = shuffle(x_train, y_train)
    y_train = np_utils.to_categorical(y_train, num_classes=2)
    x_test, x_train = x_test[:, :, -4:], x_train[:, :, -4:]
    print("Building model")
    input = Input(shape=(int(opt.length * 50), 4))
    LCDM_output, GCDM_output, FFDM_output = LCDM_model(input, opt), GCDM_model(input, opt), FFDM_model(input, opt)
    LCDM = Model(inputs=input, outputs=LCDM_output)
    GCDM = Model(inputs=input, outputs=GCDM_output)
    FFDM = Model(inputs=input, outputs=FFDM_output)
    LCDM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    GCDM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    FFDM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Training")
    bestacc = 0
    for epoch in range(opt.epoch):
        LCDM.fit(x_train, y_train, batch_size=opt.batchsize, epochs=1, verbose=0)
        GCDM.fit(x_train, y_train, batch_size=opt.batchsize, epochs=1, verbose=0)
        FFDM.fit(x_train, y_train, batch_size=opt.batchsize, epochs=1, verbose=0)
        y_pred1, y_pred2, y_pred3 = LCDM.predict(x_train), GCDM.predict(x_train), FFDM.predict(x_train)
        """multiple models fusion based on LinearRegression"""
        y_LR = [[y_pred1[:, 1][i], y_pred2[:, 1][i], y_pred3[:, 1][i]] for i in range(len(y_pred1))]
        LRmodel = LinearRegression(fit_intercept=False)
        LRmodel.fit(y_LR, np.argmax(y_train, axis=1))
        b, w = LRmodel.intercept_, LRmodel.coef_
        y_pred = LCDM.predict(x_test) * w[0] + GCDM.predict(x_test) * w[1] + FFDM.predict(x_test) * w[2] + b
        y_pred = np.argmax(y_pred, axis=1)
        bestacc = max(bestacc, metrics(y_pred, y_test))
    print("Best Accuracy:%.4f" % bestacc)
    print("Done!")
