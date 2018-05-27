# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import tensorflow as tf
import numpy as np
import csv
import random

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
        with open(filepath+"numerai_tournament_data.csv", "r") as f:
            reader = csv.reader(f)
            rs = [row for row in reader]
            rows += rs[1:]
        with open(filepath+"numerai_training_data.csv", "r") as f:
            reader = csv.reader(f)
            rs = [row for row in reader]
            rows += rs[1:]
           
        i = 0
        for r in rows:
            #r = rows[i]
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
        for i in xrange(batchsize):
            k = random.randint(0,len(self.trainset_idx)-1)
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
            

def inference(inp, m):
    #data shape : inp -- bs, m
    layerdims = [128,64,32]
    vec = inp
    lastdim = m
    cnt = 0
    for dim in layerdims:
        cnt += 1
        print cnt
        w = tf.Variable(name='w_'+str(cnt),initial_value=tf.random_normal([lastdim, dim]))
        b = tf.Variable(name='b_'+str(cnt),initial_value=tf.random_normal([dim]))
        vec = tf.matmul(vec,w) + b
        vec = tf.nn.relu(vec)
        lastdim = dim
    w = tf.Variable(name='w_'+str(cnt),initial_value=tf.random_normal([lastdim, 1]))
    b = tf.Variable(name='b_'+str(cnt),initial_value=tf.random_normal([1]))
    vec = tf.matmul(vec,w) + b
    return vec

if __name__=='__main__':
    m = 50
    batchsize = 5000
    data = tf.placeholder(dtype=tf.float32, shape=[None, m])
    label = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    logit = inference(data, m)
    
    sigmoid = tf.sigmoid(logit)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logit,labels = label))
    trainop = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.99).minimize(loss)
    dataset = Dataset('../data/numerai_datasets/')
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(2000):
            traindata, trainlabel = dataset.get_train_batch(batchsize)
            tloss,_ = sess.run([loss,trainop],feed_dict={data:traindata,label:trainlabel})
            print i,tloss
            if i%20 == 19:
                traindata, trainlabel = dataset.get_testset()
                tloss = sess.run([loss],feed_dict={data:traindata,label:trainlabel})
                print "[test] set loss:",tloss
            