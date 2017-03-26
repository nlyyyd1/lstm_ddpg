#! /usr/bin/env python
#coding=utf-8
import numpy as np
from actor_net import ActorNet
from critic_net import CriticNet

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
        
        if is_batch_norm:
            self.critic_net = CriticNet(self.num_states,self.num_actions)
            self.actor_net = ActorNet(self.num_states,self.num_actions)
        else:
            self.critic_net = CriticNet(self.num_states,self.num_actions)
            self.actor_net = ActorNet(self.num_states,self.num_actions)
        
        self.replay_memory = deque()
        
        self.time_step = 0
        self.conter = 0
        
        #因为是连续的，刚开始要确定action是多大，范围是多少。
        action_max = np.array(env.action_space.high).tolist()
        action_min = np.array(env.action_space.low).tolist()
        action_bounds = [action_max,action_min]
        
        #初始化一个东西，计算Q的梯度用的
        self.grad_inv = grad_inverter(action_bounds)
     
    #用来输出action用的
    def evaluate_actor(self,state_t):
        return self.actor_net.evaluate_actor(state_t)
    
    #用来初始化batch，然而我没有见过调用过。。。。
    def init_experience(self,file_name):
        self.replay_memory = pickle.load(open(file_name,'r'))
    
    #用来休整记忆
    def add_experience(self,observation_1,observation_2,action,reward,done):
        self.observation_1 = observation_1
        self.observation_2 = observation_2
        self.action = action
        self.reward = reward
        self.done = done
        self.replay_memory.append((self.observation_1,self.observation_2,self.action,self.reward,self.done))
        self.time_step = self.time_step + 1
        #这个timestep跟情节无关，可以一直加加加
        if(len(self.replay_memory)>REPLAY_MEMORY_SIZE):
            self.replay_memory.popleft()
        #记忆是一个队列，当长度用完了，就把最开始的给丢掉就行了
            
    def minibatchs(self):
        batch = random.sample(self.replay_memory,BATCH_SIZE)
        self.state_t_batch = [item[0] for item in batch]
        self.state_t_batch = np.array(self.state_t_batch)
        self.state_t_1_batch = [item[1] for item in batch]
        self.state_t_1_batch = np.array(self.state_t_1_batch)
        self.action_batch = [item[2] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(self.action_batch,[len(self.action_batch),self.num_actions])
        #都是列向量，不知道这样是为啥，上一个就已经是列向量
        self.reward_batch = [item[3] for item in batch]
        self.reward_batch = np.array(self.reward_batch)
        self.done_batch = [item[4] for item in batch]
        self.done_batch = np.array(self.done_batch)
        
        #state_t_batch 是所有sample的值，是一个列向量
        
    def train(self):
        self.minibatchs()
        #先拿到一个batch
        self.action_t_1_batch = self.actor_net.evaluate_target_actor(self.state_t_1_batch)
#       #下一个时刻的action倒是用target的actor网络计算出来的
        
        q_t_1 = self.critic_net.evaluate_target_critic(self.state_t_1_batch,self.action_t_1_batch)
        #下一个时刻的value值也是通过target的critic网络计算出来的
        
        self.y_i_batch = []#我们根据target网络求得时间差分的前一项rt+1+gamma*qt+1
        for i in range(0,BATCH_SIZE):
            if self.done_batch[i]:
                self.y_i_batch.append(self.reward_batch[i])
            else:
                self.y_i_batch.append(self.reward_batch[i]+GAMMA*q_t_1[i][0])
        self.y_i_batch = np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch,[len(self.y_i_batch),1])#我感觉就是为了防止他不是列向量
        
        #用target网络求出来时间差分钱项y之后，就可以更新q函数的网络了。
        self.critic_net.train_critic(self.state_t_batch,self.action_batch,self.y_i_batch)
        
        #更新完q后，要更新actor需要选中a的q的梯度，用actor网络选a
        action_for_delQ = self.evaluate_actor(self.state_t_batch)
        
        if is_grad_inverter:
            #求梯度
            self.detq = self.critic_net.compute_delQ_a(self.state_t_batch,action_for_delQ)
            self.detq = self.grad_inv.invert(self.detq,action_for_delQ)
        else:
            self.detq = self.critic_net.compute_delQ_a(self.action_t_batch,action_for_delQ)[0]
        
        #将梯度带入actor网络进行训练
        self.actor_net.train_actor(self.state_t_batch,self.detq)
        
        #训练好后要更新target网络了
        self.critic_net.update_target_critic()
        
        self.actor_net.update_target_actor()
        
    def save(self,actor_path,critic_path):
        self.critic_net.save_critic(critic_path)
        self.actor_net.save_actor(actor_path)
        
    def load(self,actor_path,critic_path):
        self.critic_net.load_critic(critic_path)
        self.actor_net.load_actor(actor_path)