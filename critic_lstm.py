#! /usr/bin/env python
#coding=utf-8
#! /usr/bin/env python
#coding=utf-8

import numpy as np
import tensorflow as tf
import math

LEARNING_RATE = 0.0001
BATCH_SIZE = 1
TAU = 0.5

class CriticNet:
    '''
    actor network model of DDPG algorthm
    '''
    
    def __init__(self,num_states,num_actions):
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            #critic network
            self.w1,self.b1,self.w2,self.w2_action,self.b2,self.w3,self.b3,self.critic_q_model,self.critic_state_in,self.critic_action_in=self.create_critic_net(num_states,num_actions)
            
            #target network
            self.t_w1,self.t_b1,self.t_w2,self.t_w2_action,self.t_b2,self.t_w3,self.t_b3,self.t_critic_q_model,self.t_critic_state_in,self.t_critic_action_in=self.create_critic_net(num_states,num_actions)
            
            #cost
            self.q_value_in = tf.placeholder('float',[None,1])
            self.l2_regularizer_loss = 0.0001*tf.reduce_sum(tf.pow(self.w2,2))+0.0001*tf.reduce_sum(tf.pow(self.b2,2))
            self.cost = tf.pow(self.critic_q_model-self.q_value_in,2)/BATCH_SIZE +self.l2_regularizer_loss
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
            
            self.act_grad_v = tf.gradients(self.critic_q_model,self.critic_action_in)
            self.action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])]#这是归一了？
            #跟batch有关？
            self.check_fl = self.action_gradients
                        
            self.update_target_w1 = tf.assign(self.t_w1,TAU*self.w1+(1-TAU)*self.t_w1)
            self.update_target_b1 = tf.assign(self.t_b1,TAU*self.b1+(1-TAU)*self.t_b1)
            self.update_target_w2 = tf.assign(self.t_w2,TAU*self.w2+(1-TAU)*self.t_w2)
            self.update_target_w2_action = tf.assign(self.t_w2_action,TAU*self.w2_action+(1-TAU)*self.t_w2_action)
            self.update_target_b2 = tf.assign(self.t_b2,TAU*self.b2+(1-TAU)*self.t_b2)
            self.update_target_w3 = tf.assign(self.t_w3,TAU*self.w3+(1-TAU)*self.t_w3)
            self.update_target_b3 = tf.assign(self.t_b3,TAU*self.b3+(1-TAU)*self.t_b3)
            
            self.saver = tf.train.Saver()
            self.sess.run(tf.initialize_all_variables())
            
            self.sess.run([
                self.t_w1.assign(self.w1),
                self.t_w2.assign(self.w2),
                self.t_w2_action.assign(self.w2_action),
                self.t_w3.assign(self.w3),
                self.t_b1.assign(self.b1),
                self.t_b2.assign(self.b2),
                self.t_b3.assign(self.b3),])
                
    def create_critic_net(self,num_states,num_actions):
        '''
        network that takes states and return action
        '''
        N_HIDDEN_1 = 40
        N_HIDDEN_2 = 30
        critic_state_in = tf.placeholder('float',[None,num_states])
        critic_action_in = tf.placeholder('float',[None,num_actions])
        w1=tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        w2=tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))
        w2_action=tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))
        w3=tf.Variable(tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003))
        b1=tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        b2=tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_states),1/math.sqrt(N_HIDDEN_1+num_states)))
        b3=tf.Variable(tf.random_uniform([1],-0.003,0.003))
                        
        h1 = tf.nn.relu(tf.matmul(critic_state_in,w1)+b1)
        h2 = tf.nn.relu(tf.matmul(h1,w2)+tf.matmul(critic_action_in,w2_action)+b2)
        critic_q_model = tf.matmul(h2,w3)+b3
        return w1,b1,w2,w2_action,b2,w3,b3,critic_q_model,critic_state_in,critic_action_in    
    
    def evaluate_critic(self,state_t,action_t):
        #这个函数是用来给state和action来得到q函数的
        return self.sess.run(self.critic_q_model,feed_dict={self.critic_state_in:stete_t,self.critic_aciton_in:action_t})
    
    def evaluate_target_critic(self,state_t_1,action_t_1):
        #这个是target网络的
        return self.sess.run(self.t_critic_q_model,feed_dict={self.t_critic_state_in:state_t_1,self.t_critic_action_in:action_t_1})
    
    def train_critic(self,state_t_batch,action_batch,y_i_batch):
        self.sess.run(self.optimizer,feed_dict={self.critic_state_in:state_t_batch,self.critic_action_in:action_batch,self.q_value_in:y_i_batch})
        
    def update_target_critic(self):
        self.sess.run([self.update_target_w1,self.update_target_b1,
                       self.update_target_w2,self.update_target_b2,
                       self.update_target_w2_action,
                       self.update_target_w3,self.update_target_b3])
    
    def save_actor(self,path_name):
        save_path = self.saver.save(self.sess,path_name)
        print "Critic model saved in file %s" % save_path
        
    def load_actor(self,path_name):
        self.saver.restore(self.sess,path_name)
        print "Critic model restored"
        
    def compute_delQ_a(self,state_t,action_t):
        return self.sess.run(self.action_gradients,feed_dict={self.critic_state_in:state_t,self.critic_action_in:action_t})
        
