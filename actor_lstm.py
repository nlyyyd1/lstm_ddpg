#! /usr/bin/env python
#coding=utf-8

import numpy as np
import tensorflow as tf
import math

from tensorflow.models.rnn.ptb import reader

LEARNING_RATE = 0.0001
BATCH_SIZE = 1
TAU = 0

class ActorNet:
    '''
    actor network model of DDPG algorthm
    '''
    
    def __init__(self,num_states,num_hidden_states,num_actions):
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            #actor network
            self.actor_parameters,self.actor_model,self.input,self.output,self.hiddenc,self.statec,self.stateh = self.create_actor(num_states,num_hidden_states,num_actions,'actor')
            self.target_actor_parameters,self.target_actor_model,self.target_input,self.target_output,self.target_hiddenc,self.target_statec,self.target_stateh = self.create_actor(num_states,num_hidden_states,num_actions,'target')
            
            self.target_actor_parameters = self.target_actor_parameters[4:]
            #cost
            self.q_gradient_input = tf.placeholder('float',[None,num_actions])
            gradients = tf.gradients(self.actor_model,self.actor_parameters,-self.q_gradient_input)
            #tf.gradient(x,y,z)表示的是求z×dx/dy
            self.parameters_gradients,_ =tf.clip_by_global_norm(gradients,5) 
            #要控制一下梯度膨胀
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.actor_parameters))
            
            self.update_target_parameters0 = tf.assign(self.target_actor_parameters[0],TAU*self.target_actor_parameters[0]+(1-TAU)*self.actor_parameters[0])
            self.update_target_parameters1 = tf.assign(self.target_actor_parameters[1],TAU*self.target_actor_parameters[1]+(1-TAU)*self.actor_parameters[1])
            self.update_target_parameters2 = tf.assign(self.target_actor_parameters[2],TAU*self.target_actor_parameters[2]+(1-TAU)*self.actor_parameters[2])
            self.update_target_parameters3 = tf.assign(self.target_actor_parameters[3],TAU*self.target_actor_parameters[3]+(1-TAU)*self.actor_parameters[3])
                            
            self.saver = tf.train.Saver()
            self.sess.run(tf.initialize_all_variables())
            
            self.sess.run(self.target_actor_parameters[0].assign(self.actor_parameters[0]))
            self.sess.run(self.target_actor_parameters[1].assign(self.actor_parameters[1]))
            self.sess.run(self.target_actor_parameters[2].assign(self.actor_parameters[2]))
            self.sess.run(self.target_actor_parameters[3].assign(self.actor_parameters[3]))
                                    
            
    def create_actor(self,num_states,num_hidden_states,num_actions,models):
        
        input = tf.placeholder(tf.float32,[BATCH_SIZE,num_states])
        statec = tf.placeholder(tf.float32,[BATCH_SIZE,num_hidden_states])
        stateh = tf.placeholder(tf.float32,[BATCH_SIZE,num_hidden_states])
        state = tuple([statec,stateh])        
                        
        if models == 'actor':
            with tf.variable_scope('actor'):
                actor_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_states,forget_bias = 1.0,state_is_tuple = True)
                #state = actor_cell.zero_state(BATCH_SIZE,tf.float32)
                (output,state) = actor_cell(input,state)
        if models == 'target':
            with tf.variable_scope('target'):
                target_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_states,forget_bias = 1.0,state_is_tuple = True)
                #state = target_cell.zero_state(BATCH_SIZE,tf.float32)
                (output,state) = target_cell(input,state) 
        #防止output和state超过范围
        output = tf.reshape(output,[-1,num_hidden_states])
        #state 和output超范围应该是参数超范围了，所以应该改参数
        #output = tf.clip_by_norm(output,math.sqrt(num_hidden_states))
        #s = [[1],[1]]
        #s[0] = tf.clip_by_norm(state[0],math.sqrt(num_hidden_states))
        #s[1] = tf.clip_by_norm(state[1],math.sqrt(num_hidden_states)) 
        #state = ([s[0],s[1]])   
    
        w=tf.Variable(tf.random_uniform([num_hidden_states,num_actions],-0.0003,0.0003))
        b=tf.Variable(tf.random_uniform([num_actions],-0.0003,0.0003))
                    
        actor_model = tf.matmul(output,w)+b
        #actor_model = tf.clip_by_value(actor_model,-1,1)
        actor_parameters = tf.trainable_variables()
        return actor_parameters,actor_model,input,state[1],state[0],statec,stateh
        
            
         
    def evaluate_actor(self,state_t,c,h):
        #这个函数是用来给state来得到actor的，actor用actor_model来表示
        return self.sess.run([self.actor_model,self.hiddenc,self.output],feed_dict={self.input:state_t,self.statec:c,self.stateh:h})

    def evaluate_target_actor(self,state_t,c,h):
        return self.sess.run([self.target_actor_model,self.target_hiddenc,self.target_output],feed_dict={self.target_input:state_t,self.target_statec:c,self.target_stateh:h})
    
    def train_actor(self,actor_state_in,q_gradient_input,c,h):
        return self.sess.run([self.hiddenc,self.output,self.optimizer],feed_dict={self.input:actor_state_in,self.q_gradient_input:q_gradient_input,self.statec:c,self.stateh:h})

    def update_target_actor(self):
        self.sess.run([self.update_target_parameters0,self.update_target_parameters1,self.update_target_parameters2,self.update_target_parameters3])
        
    def save_actor(self,path_name):
        save_path = self.saver.save(self.sess,path_name)
        print "Actor model saved in file %s" % save_path
        
    def load_actor(self,path_name):
        self.saver.restore(self.sess,path_name)
        print "Actor model restored"
        
