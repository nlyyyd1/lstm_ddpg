#! /usr/bin/env python
#coding=utf-8

import numpy as np
import tensorflow as tf
import math


LEARNING_RATE = 0.0001
BATCH_SIZE = 1
size = 3
num_states = 1
num_actions = 1

with tf.Graph().as_default():
    sess = tf.InteractiveSession()
    cell = tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias = 0.0,state_is_tuple = True)
    input_data = tf.placeholder(tf.float32,[1,num_states])
    output_data = tf.placeholder(tf.float32,[1,num_actions])
    c = tf.Variable(tf.zeros([1,3]))
    h = tf.Variable(tf.zeros([1,3]))
 
    statec = tf.placeholder(tf.float32,[1,size])
    stateh = tf.placeholder(tf.float32,[1,size])
    state = tuple([statec,stateh])

    input = input_data
    print state
    (output,state) = cell(input,state)
    output = tf.reshape(output,[-1,size])
    print state

    w = tf.Variable(tf.truncated_normal([size,num_actions],stddev=0.1))
    
    b = tf.Variable(tf.truncated_normal([num_actions],stddev=0.1))

    logist = tf.matmul(output,w)+b
    
    loss = -tf.reduce_sum(output_data*tf.log(logist))
    
    train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    sess.run(tf.initialize_all_variables())
    
    c = np.array([[0,0,0]])
    h = np.array([[0,0,0]]) 
    s = ([c,h])   
    
    for i in range(9):
        x = np.array(1).reshape(1,1)
        y = np.array(1).reshape(1,1)
        
       
        if i == 0:
            [o,s[0],s[1]] = sess.run([output,state[0],state[1]],feed_dict={input_data:x,output_data:y,stateh:h,statec:c})
            print o
            print s[1]
        else:
            [o,s[0],s[1]] = sess.run([output,state[0],state[1]],feed_dict={input_data:x,output_data:y,stateh:s[1],statec:s[0]})
            print o            
        


        #已经解决了state赋值的问题，
        #未解决如何同事计算state和output的问题,
        
        #这些问题的关键都在与tuple与tensor不好转换，将tuple分开，问题就迎刃而解了
        
   