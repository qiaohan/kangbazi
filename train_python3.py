# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
import numpy as np
import csv
import random
import os
from sklearn import metrics
import pickle
import math



class Dataset(object):
    def __init__(self, filepath):
        self.features = []
        self.targets = []
        self.ids = []
        self.era = []
        self.data_type = []
        self.trainset_idx = []
        self.testset_idx = []

        rows = []
        tournament_path = os.path.join(filepath, 'numerai_tournament_data.csv')
        with open(tournament_path, "r") as f:
            reader = csv.reader(f)
            rs = [row for row in reader]
            rows += rs[1:]

        training_path = os.path.join(filepath, "numerai_training_data.csv")
        with open(training_path, "r") as f:
            reader = csv.reader(f)
            rs = [row for row in reader]
            rows += rs[1:]

        i = 0
        for r in rows:
            # r = rows[i]
            if not r[2] in ['train', 'validation']:
                continue
            self.features.append([float(s) for s in r[3:-1]])
            self.targets.append([float(r[-1])])
            self.ids.append(r[0])
            self.era.append(r[1])
            self.data_type.append(r[2])
            if r[2] == 'validation':
                self.testset_idx.append(i)
            elif r[2] == 'train':
                self.trainset_idx.append(i)
            i += 1

    def get_train_batch(self, batchsize):
        fea = []
        lab = []
        for i in range(batchsize):
            k = random.randint(0, len(self.trainset_idx) - 1)
            kk = self.trainset_idx[k]
            fea.append(self.features[kk])
            lab.append(self.targets[kk])
        return np.asarray(fea), np.asarray(lab)

    def get_testset(self):
        if not hasattr(self, 'testfea'):
            self.testfea = []
            self.testlab = []
            for kk in self.testset_idx:
                self.testfea.append(self.features[kk])
                self.testlab.append(self.targets[kk])
        return np.asarray(self.testfea), np.asarray(self.testlab)


def inference(inp, m, is_training_flag):
    # is_training_flag is a place holder

    # data shape : inp -- bs, m
    layerdims = [128, 64, 32]
    vec = inp
    lastdim = m
    cnt = 0
    for dim in layerdims:
        cnt += 1
        print(cnt)
        # w = tf.Variable(name='w_' + str(cnt), initial_value = tf.contrib.layers.xavier_initializer([lastdim, dim]))
        # b = tf.Variable(name='b_' + str(cnt), initial_value = tf.zeros([dim]))
        w = tf.get_variable(name='w_' + str(cnt), shape = [lastdim, dim], initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b_' + str(cnt), shape = [dim], initializer = tf.zeros_initializer())
        vec = tf.matmul(vec, w) + b
        vec = tf.contrib.layers.batch_norm(vec, center=False, scale=False, is_training = is_training_flag)

        vec = tf.nn.relu(vec)
        lastdim = dim

    # w = tf.Variable(name='w_' + str(cnt), initial_value = tf.contrib.layers.xavier_initializer([lastdim, 1]))
    # b = tf.Variable(name='b_' + str(cnt), initial_value = tf.zeros([1]))
    cnt += 1
    w = tf.get_variable(name='w_' + str(cnt), shape = [lastdim, 1], initializer = tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name='b_' + str(cnt), shape = [1], initializer = tf.zeros_initializer())

    vec = tf.matmul(vec, w) + b
    return vec

if __name__ == '__main__':
    m = 50
    batchsize = 5000
    train_step_max = 5000
    n_eval_step = 100
    learning_rate_bank = [1e-4, 1e-5, 1e-6]
    learning_step_bank = [1000, 2000, 3000]


    data = tf.placeholder(dtype=tf.float32, shape=[None, m])
    label = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(dtype = tf.float32)
    is_training_flag = tf.placeholder(dtype=tf.bool)
    logit = inference(data, m, is_training_flag)

    sigmoid = tf.sigmoid(logit)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=label))
    trainop = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum=0.99).minimize(loss)


    dataset = Dataset(os.path.join(os.getcwd(), 'data', 'numerai_data_sets'))
    auc = np.zeros(math.ceil(train_step_max // n_eval_step))

    which_rate = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        for i in range(train_step_max):
            traindata, trainlabel = dataset.get_train_batch(batchsize)

            if (which_rate < len(learning_step_bank) - 1) and i == learning_step_bank[which_rate]:
                which_rate += 1
                print('learning rate is {}'.format(learning_rate_bank[which_rate]))

            tloss, _ = sess.run([loss, trainop],
                                feed_dict={
                                 data: traindata,
                                 label: trainlabel,
                                 learning_rate: learning_rate_bank[which_rate],
                                 is_training_flag: True
                                 })
            # print(i, tloss)


            if i % n_eval_step == (n_eval_step - 1):
                traindata, trainlabel = dataset.get_testset()
                tloss, tsigmoid, tlogit = sess.run([loss,  sigmoid, logit],
                                                   feed_dict={data: traindata, label: trainlabel, is_training_flag:False})
                print("[test] set loss:", tloss)

                fpr, tpr, thresh = metrics.roc_curve(trainlabel, tsigmoid)
                auc[i // n_eval_step] = metrics.auc(fpr, tpr)



            # if (i % 500 == 499 or i == train_step_max):
            #
            #     check_point_path = os.path.join(os.getcwd(), 'train', 'model_v0.cpkt')
            #     saver.save(sess, check_point_path, global_step=i)

        ## separate the training and evaluation in the future.

        ## save the imtermediate training data
        # data_path = os.path.join(os.getcwd(), 'data_tmp', 'roc_model_v0')
        # with open(data_path, 'wb') as f:
        #     data_roc = dict()
        #     data_roc['auc'] = auc
        #     data_roc['final_fpr'] = fpr
        #     data_roc['final_tpr'] = tpr
        #     pickle.dump(data_roc, f)