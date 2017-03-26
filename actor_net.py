#! /usr/bin/env python
#coding=utf-8

import numpy as np
import tensorflow as tf
import math

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
TAU = 0.5

class ActorNet:
    '''
    actor network model of DDPG algorthm
    '''
    
    def __init__(self,num_states,num_actions):
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            #actor network
            self.w1,self.b1,self.w2,self.b2,self.w3,self.b3,self.actor_state_in,self.actor_model=self.create_actor_net(num_states,num_actions)
            
            #target network
            self.t_w1,self.t_b1,self.t_w2,self.t_b2,self.t_w3,self.t_b3,self.t_actor_state_in,self.t_actor_model=self.create_actor_net(num_states,num_actions)
            
            #cost
            self.q_gradient_input = tf.placeholder('float',[None,num_actions])
            self.actor_parameters = [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]
            self.parameters_gradients = tf.gradients(self.actor_model,self.actor_parameters,-self.q_gradient_input)
            #tf.gradient(x,y,z)表示的是求z×dx/dy
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.actor_parameters))
        
            
            self.update_target_w1 = tf.assign(self.t_w1,TAU*self.w1+(1-TAU)*self.t_w1)
            self.update_target_b1 = tf.assign(self.t_b1,TAU*self.b1+(1-TAU)*self.t_b1)
            self.update_target_w2 = tf.assign(self.t_w2,TAU*self.w2+(1-TAU)*self.t_w2)
            self.update_target_b2 = tf.assign(self.t_b2,TAU*self.b2+(1-TAU)*self.t_b2)
            self.update_target_w3 = tf.assign(self.t_w3,TAU*self.w3+(1-TAU)*self.t_w3)
            self.update_target_b3 = tf.assign(self.t_b3,TAU*self.b3+(1-TAU)*self.t_b3)
            
            self.saver = tf.train.Saver()
            self.sess.run(tf.initialize_all_variables())
            
            self.sess.run([
                self.t_w1.assign(self.w1),
                self.t_w2.assign(self.w2),
                self.t_w3.assign(self.w3),
                self.t_b1.assign(self.b1),
                self.t_b2.assign(self.b2),
                self.t_b3.assign(self.b3),])
                
    def create_actor_net(self,num_states=4,num_actions=1):
        '''
        network that takes states and return action
        '''
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        actor_state_in = tf.placeholder('float',[None,num_states])
        w1=tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        w2=tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        w3=tf.Variable(tf.random_uniform([N_HIDDEN_2,num_actions],-1/math.sqrt(N_HIDDEN_2),1/math.sqrt(N_HIDDEN_2)))
        b1=tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        b2=tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        b3=tf.Variable(tf.random_uniform([num_actions],-1/math.sqrt(N_HIDDEN_2),1/math.sqrt(N_HIDDEN_2)))
                        
        h1 = tf.nn.relu(tf.matmul(actor_state_in,w1)+b1)
        h2 = tf.nn.relu(tf.matmul(h1,w2)+b2)
        actor_model = tf.matmul(h2,w3)+b3
        return w1,b1,w2,b2,w3,b3,actor_state_in,actor_model    
    
    def evaluate_actor(self,state_t):
        #这个函数是用来给state来得到actor的，actor用actor_model来表示
        return self.sess.run(self.actor_model,feed_dict={self.actor_state_in:state_t})
    
    def evaluate_target_actor(self,state_t_1):
        #这个是target网络的
        return self.sess.run(self.t_actor_model,feed_dict={self.t_actor_state_in:state_t_1})
    
    def train_actor(self,actor_state_in,q_gradient_input):
        self.sess.run(self.optimizer,feed_dict={self.actor_state_in:actor_state_in,self.q_gradient_input:q_gradient_input})
        
    def update_target_actor(self):
        self.sess.run([self.update_target_w1,self.update_target_b1,
                       self.update_target_w2,self.update_target_b2,
                       self.update_target_w3,self.update_target_b3])
    
    def save_actor(self,path_name):
        save_path = self.saver.save(self.sess,path_name)
        print "Actor model saved in file %s" % save_path
        
    def load_actor(self,path_name):
        self.saver.restore(self.sess,path_name)
        print "Actor model restored"
        