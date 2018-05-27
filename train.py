# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import tensorflow as tf
import numpy as np

class Dataset(object):
    def __init__(self, filepath):
        self.


def inference(inp, m):
    #data shape : inp -- bs, m
    layerdims = [128,64,32]
    vec = inp
    lastdim = m
    for dim in layerdims:
        w = tf.get_variable(shape=[lastdim,dim], initializer=tf.in)
        b = tf.get_variable(shape=[dim])
        vec = tf.matmul(vec,w) + b
        vec = tf.nn.relu(vec)
        lastdim = dim
    w = tf.get_variable(shape=[lastdim,1], initializer=tf.in)
    b = tf.get_variable(shape=[1])
    vec = tf.matmul(vec,w) + b
    return vec

if __name__=='__main__':
    m = 50
    data = tf.placeholder(dtype=tf.float32, shape=[None, m])
    label = tf.placeholder(dtype=tf.float32, shape=[None,1])
    logit = inference(data, m)
    
    sigmoid = tf.sigmoid(logit)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logit,labels = label))
    
    dataset = Dataset('../data/numerai_datasets/')
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(20):
            