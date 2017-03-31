#! /usr/bin/env python
#coding=utf-8
import numpy as np
from actor_lstm import ActorNet
from critic_lstm import CriticNet

from collections import deque
from gym.spaces import Box,Discrete
import random
from tensorflow_grad_inverter import grad_inverter
import time
import pickle

REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
is_grad_inverter = True

class DDPG:
    '''
    deep deterministic policy gradient algorithm
    '''
    #初始化一个ddpg，基本就是定义好state，action，因为action是输出，还要定义好他的大小范围，初始化两个网络类    
    def __init__(self,env,is_batch_norm):
        self.env = env
        self.num_states = env.observation_space.shape[0]-1
        self.num_actions = env.action_space.shape[0]
        self.num_hidden_states = 5
        
        if is_batch_norm:
            self.critic_net = CriticNet(self.num_states,self.num_hidden_states,sself.num_actions)
            self.actor_net = ActorNet(self.num_states,self.num_hidden_states,self.num_actions)
        else:
            self.critic_net = CriticNet(self.num_states,self.num_hidden_states,self.num_actions)
            self.actor_net = ActorNet(self.num_states,self.num_hidden_states,self.num_actions)
        
        #因为是连续的，刚开始要确定action是多大，范围是多少。
        action_max = np.array(env.action_space.high).tolist()
        action_min = np.array(env.action_space.low).tolist()
        action_bounds = [action_max,action_min]
        
        #初始化一个东西，计算Q的梯度用的
        self.grad_inv = grad_inverter(action_bounds)
        self.replay_memory = deque()
     
    #用来输出action用的
    def evaluate_actor(self,state_t,c,h):
        return self.actor_net.evaluate_actor(state_t,c,h)
    
    def init_experience(self,file_name):
        self.replay_memory = pick.load(open(file_name,'r'))
    
    def add_experience(self,statec_t1,stateh_t1,ob_t,action_t,ob_t_1,reward_t_1,done):
        self.replay_memory.append((statec_t1,stateh_t1,ob_t,action_t,ob_t_1,reward_t_1,done))
        if(len(self.replay_memory)>REPLAY_MEMORY_SIZE):
            self.replay_memory.popleft()
            
    def minibatches(self):
        batch = random.sample(self.replay_memory,BATCH_SIZE)
        self.statec_t1 = [item[0] for item in batch]
        self.statec_t1 = np.array(self.statec_t1).reshape(BATCH_SIZE,-1)
        self.stateh_t1 = [item[1] for item in batch]
        self.stateh_t1 = np.array(self.stateh_t1).reshape(BATCH_SIZE,-1)#这个是output
        self.obser_t   = [item[2] for item in batch]
        self.obser_t   = np.array(self.obser_t).reshape(BATCH_SIZE,-1)
        self.action_t  = [item[3] for item in batch]
        self.action_t  = np.array(self.action_t).reshape(BATCH_SIZE,-1)
        self.obser_t_1 = [item[4] for item in batch]
        self.obser_t_1 = np.array(self.obser_t_1).reshape(BATCH_SIZE,-1)
        self.reward_t_1= [item[5] for item in batch]
        self.reward_t_1= np.array(self.reward_t_1).reshape(BATCH_SIZE,-1)
        self.done      = [item[6] for item in batch]
        self.done      = np.array(self.done).reshape(BATCH_SIZE,-1)        
        
    
    
    def train(self):
        self.minibatches()
        
        _,self.statec_t,self.stateh_t = self.actor_net.evaluate_actor(self.obser_t,self.statec_t1,self.stateh_t1)
        self.action_t_1,self.statec_t_1,self.stateh_t_1 = self.actor_net.evaluate_target_actor(self.obser_t_1,self.statec_t,self.stateh_t)
        #下一个时刻的action倒是用target的actor网络计算出来的
        
        self.stateh_t_1 = self.stateh_t_1.reshape(BATCH_SIZE,-1)
        self.action_t_1 = self.action_t_1.reshape(BATCH_SIZE,-1)
        
        state_t_1 = np.hstack([self.obser_t_1,self.stateh_t_1])
        state_t_1 = state_t_1.reshape(BATCH_SIZE,-1)
        
        q_t_1 = self.critic_net.evaluate_target_critic(state_t_1,self.action_t_1)
        #下一个时刻的value值也是通过target的critic网络计算出来的
        
        self.y_i = []#我们根据target网络求得时间差分的前一项rt+1+gamma*qt+1
        for i in range(0,BATCH_SIZE):
            if self.done[i]:
                self.y_i.append(self.reward_t_1[i])
            else:
                self.y_i.append(self.reward_t_1[i]+GAMMA*q_t_1[i][0])
        self.y_i = np.array(self.y_i)
        self.y_i = np.reshape(self.y_i,[len(self.y_i),1])#我感觉就是为了防止他不是列向量
        
        #用target网络求出来时间差分钱项y之后，就可以更新q函数的网络了。
        state_t = np.hstack([self.obser_t,self.stateh_t1])
        state_t = state_t.reshape(BATCH_SIZE,-1)
        self.critic_net.train_critic(state_t,self.action_t,self.y_i)
        
        action_for_delQ,_,_ = self.evaluate_actor(self.obser_t,self.statec_t1,self.stateh_t1)
        #action_for_delQ = np.array([action_for_delQ]).reshape(BATCH_SIZE,-1)
        
        if is_grad_inverter:
            #求梯度
            self.detq = self.critic_net.compute_delQ_a(state_t,action_for_delQ)
            self.detq = self.grad_inv.invert(self.detq,action_for_delQ)
        else:
            self.detq = self.critic_net.compute_delQ_a(state_t,action_for_delQ)[0]
        
        #将梯度带入actor网络进行训练
        self.actor_net.train_actor(self.obser_t,self.detq,self.statec_t1,self.stateh_t1)
        #不用target网络的，用真实网络的试一试，target只用来计算label
        #训练好后要更新target网络了
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()
        
    def save(self,actor_path,critic_path):
        self.critic_net.save_critic(critic_path)
        self.actor_net.save_actor(actor_path)
        
    def load(self,actor_path,critic_path):
        self.critic_net.load_critic(critic_path)
        self.actor_net.load_actor(actor_path)
