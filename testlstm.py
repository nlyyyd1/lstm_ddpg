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
    state = cell.zero_state(1,tf.float32)
    input = input_data
    (output,state) = cell(input,state)
    output = tf.reshape(output,[-1,size])
    
    w = tf.Variable(tf.truncated_normal([size,num_actions],stddev=0.1))
    
    b = tf.Variable(tf.truncated_normal([num_actions],stddev=0.1))

    logist = tf.matmul(output,w)+b
    
    loss = -tf.reduce_sum(output_data*tf.log(logist))
    
    train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    sess.run(tf.initialize_all_variables())
    
    print state
    state_reset = tf.assign(state,([1,1,1],[2,2,2]))
    
    for i in range(3):
        x = np.array(i).reshape(1,1)
        y = np.array(i).reshape(1,1)
        print sess.run(state,feed_dict={input_data:x,output_data:y})
        sess.run(train,feed_dice={input_data:x,output_data:y})
        #print state
    print sess.run(state,feed_dict={input_data:x,output_data:y})
    sess.run(state_reset)
    print sess.run(state,feed_dict={input_data:x,output_data:y})
    


        #todo:关于state到底应该怎么办，看看不然看看feed_dict怎么搞吧，或者看看人家rnn是怎么搞的
        #经过实验，state不用feed就可以变！！！！！
        
        #下面的问题，如何让state初始化，但是其他变量不初始化呢？
        #其实还是如何手动改state的问题，哭晕