#! /usr/bin/env python
#coding=utf-8
import gym
from gym.spaces import Box,Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise

episodes = 800
is_batch_norm = False

#整体是环境交互，存储batch，最后调用train函数
#ddpg就在刚开始生成，然后每个情节，用ddpg找到action， 存起来，train一下

def main():
    experiment = 'MountainCarContinuous-v0'
    env = gym.make(experiment)
    steps = env.spec.timestep_limit
    assert isinstance(env.observation_space,Box)
    assert isinstance(env.action_space,Box)
    #这些是关于环境的，环境会定义最大步长，状态和行为的维度。状态其实是观测量
    
    agent = DDPG(env,is_batch_norm)#这个在循环前面，所以所有的weight都有继承
    #也就是说，整个过程只训练了一个模型出来。
    exploration_noise = OUNoise(env.action_space.shape[0])
    #这个探索的noise看看能不能更新升级！！！！！！！！！！！！
    counter = 0
    reward_per_episode=0
    total_reward = 0
    num_states = env.observation_space.shape[0]-1
    num_actions = env.action_space.shape[0]
    #这是state的维度和action的维度
    
    print 'Number of States:',num_states
    print 'Number of Actions:',num_actions
    print 'Number of steps per episode:',steps
    
    reward_st = np.array([0])#这个是用来存每一次的rewards的
    
    for i in xrange(episodes):#一共要循环800次
        print '====starting episode no:',i,'====','\n'
        observation = env.reset()#每个情节初始化，但是模型参数不初始化
        reward_per_episode = 0
        for t in xrange(steps):
            #env.render()
            #这里表示是否要进行渲染
            x = observation[0:num_states]
            #当前的观测变为状态为xt
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            #这里选择action用的不是target网络！
            noise = exploration_noise.noise()
            action = action[0]+noise
            #依据xt做出行为，并且行为是有一定随机性的
            #可以看看这个随机性如何更改！！！！！！！！！！！！！！！！
            #这个探索的noise只有在下一个情节的时候才更新，真的要这样么？？？？？？
            #print 'Action at step',t,':',action,'\n'
            
            observation,reward,done,info = env.step(action)
            #做完行为后得到状态xt+1,也就是新的observation
            
            agent.add_experience(x,observation[0:num_states],action,reward,done)
            #将这些东西存起来
            
            if counter>64:
                agent.train()
                #只有存满了才能开始训练，也就是说 刚开始的64步都不进行训练，他设置的batch就是64的
                #则因为 总共只跑1000步左右，会不会有什么问题？
                #不会有问题，他的counter实在最外层循环定义的，说明，整个循环套循环只用等64次初始就成了
                #所以这个train应该是batch的，能不能变成完全在线的，这样是不是网络结构要变简单才行？
            reward_per_episode += reward
            counter += 1
            #counter是为了判断经验有没有存满用的
            
            if (done or (t == steps-1)):
                #一个情节结束了～
                print 'EPISODE:',i,'Steps',t,'Total Reward:',reward_per_episode
                print 'Printing reward to file'
                exploration_noise.reset()
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st,newline='\n')
                #原来这么简单就可以吧数据给记录下来了
                print '\n\n'
                break
                
    total_reward += reward_per_episode
    #这里是计算平均值的
    print "Average reward per episode {}".format(total_reward/episodes)
    
    
if __name__=='__main__':
    main()