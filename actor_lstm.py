#! /usr/bin/env python
#coding=utf-8

import numpy as np
import tensorflow as tf
import math

from tensorflow.models.rnn.ptb import reader

LEARNING_RATE = 0.0001
BATCH_SIZE = 1
size = 100

class ActorNet:
    '''
    actor network model of DDPG algorthm
    '''
    
    def __init__(self,num_states,num_actions):
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            #actor network
            cell = tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias = 0.0,state_is_tuple = True)
            
            self.actor_state_in = tf.placeholder(tf.float32,[1,num_states])
            self._initial_state = cell.zero_state(1,tf.float32)
            input = self.actor_state_in

            (output,state) = cell(input,state)

            output = tf.reshape(output,[-1,size])
            self.output = output
            
            w=tf.Variable(tf.random_uniform([size,num_actions],-1/math.sqrt(num_actions),1/math.sqrt(num_actions)))
            b=tf.Variable(tf.random_uniform([num_actions],-1/math.sqrt(num_actions),1/math.sqrt(num_actions)))
            
            self.actor_model = tf.matmul(output,w)+b
            
            #cost
            self.q_gradient_input = tf.placeholder('float',[None,num_actions])
            self.actor_parameters = tf.trainable_variables()
            self.parameters_gradients = tf.gradients(self.actor_model,self.actor_parameters,-self.q_gradient_input)
            #tf.gradient(x,y,z)表示的是求z×dx/dy
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.actor_parameters))
    
            self.saver = tf.train.Saver()
            self.sess.run(tf.initialize_all_variables())
            
    def evaluate_actor(self,state_t):
        #这个函数是用来给state来得到actor的，actor用actor_model来表示
        return self.sess.run([self.actor_model,self.output]feed_dict={self.actor_state_in:state_t})
    
    def train_actor(self,actor_state_in,q_gradient_input):
        self.sess.run(self.optimizer,feed_dict={self.actor_state_in:actor_state_in,self.q_gradient_input:q_gradient_input})

    def save_actor(self,path_name):
        save_path = self.saver.save(self.sess,path_name)
        print "Actor model saved in file %s" % save_path
        
    def load_actor(self,path_name):
        self.saver.restore(self.sess,path_name)
        print "Actor model restored"
        
