#! /usr/bin/env python
#coding=utf-8
import gym
from gym.spaces import Box,Discrete
import numpy as np
from ddpg_lstm_new import DDPG
from ou_noise import OUNoise
import random

episodes = 1000
is_batch_norm = False
num_hidden_states = 5
BATCH_SIZE = 64
epsilon = 0.3

def main():
    experiment = 'MountainCarContinuous-v0'
    env = gym.make(experiment)
    steps = env.spec.timestep_limit
    assert isinstance(env.observation_space,Box)
    assert isinstance(env.action_space,Box)
    
    agent = DDPG(env,is_batch_norm)#这个在循环前面，所以所有的weight都有继承
    #也就是说，整个过程只训练了一个模型出来。
    exploration_noise = OUNoise(env.action_space.shape[0])
    reward_per_episode=0
    total_reward = 0
    counter = 0
    num_states = env.observation_space.shape[0]-1
    num_actions = env.action_space.shape[0]
    #这是state的维度和action的维度
    
    print 'Number of States:',num_states
    print 'Number of Actions:',num_actions
    print 'Number of steps per episode:',steps
    
    reward_st = np.array([0])#这个是用来存每一次的rewards的
    
    for i in xrange(episodes):#一共要循环1000次
        print '====starting episode no:',i,'====','\n'
        observation = env.reset()#每个情节初始化，但是模型参数不初始化
        reward_per_episode = 0
        statec = np.zeros((1,num_hidden_states))
        stateh = np.zeros((1,num_hidden_states))
        
        actor,statec_t1,stateh_t1 = agent.evaluate_actor(np.reshape(observation[0:num_states],[1,num_states]),statec,stateh)        
        action = actor[0]+exploration_noise.noise()
        observation,_,_,_ = env.step(action)
        
        for t in xrange(steps):
            #env.render()
            x = observation[0:num_states]
            actor,statec_t,stateh_t = agent.evaluate_actor(np.reshape(x,[1,num_states]),statec_t1,stateh_t1)
            noise = exploration_noise.noise()
            #ra = random.random()
            if(i<500):
                action = actor[0]+noise
            else:
                action = actor[0]
            observation,reward,done,info = env.step(action)
            #print 'Action at step',t,':',action,'reward:',reward,'\n'
            agent.add_experience(statec_t1,stateh_t1,x,action,observation[0:num_states],reward,done)
            
            if counter >64:
                agent.train()
            counter+=1
            statec_t1 = statec_t
            stateh_t1 = stateh_t
            reward_per_episode += reward
            if reward>1:
                print reward
            if reward<-1:
                print actor
                print observation
                print stateh_t
                print reward
                   
            if (done or (t == steps-1)):
                #一个情节结束了～
                print 'EPISODE:',i,'Steps',t,'Total Reward:',reward_per_episode
                print 'Printing reward to file'
                exploration_noise.reset()
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st,newline='\n')
                print '\n\n'
                break
                
    total_reward += reward_per_episode
    #这里是计算平均值的
    print "Average reward per episode {}".format(total_reward/episodes)
    
    
if __name__=='__main__':
    main()
