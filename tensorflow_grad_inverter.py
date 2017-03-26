#! /usr/bin/env python
#coding=utf-8
import tensorflow as tf

class grad_inverter:
    def __init__(self,action_bounds):
        #action_bounds = [action_max,action_min]
        self.sess = tf.InteractiveSession()
        self.action_size = len(action_bounds[0])
        #action可以是多维的
        
        self.action_input = tf.placeholder(tf.float32,[None,self.action_size])#action的batch
        self.pmax = tf.constant(action_bounds[0],dtype=tf.float32)#action每一维最大的数
        self.pmin = tf.constant(action_bounds[1],dtype=tf.float32)#action每一维最小的数
        self.prange = tf.constant([x-y for x,y in zip(action_bounds[0],action_bounds[1])],dtype=tf.float32)#acrion每一维的距离
        self.pdiff_max = tf.div(-self.action_input+self.pmax,self.prange)#该行为与最大值的距离除以range，相当于一个比例。
        self.pdiff_min = tf.div(self.action_input-self.pmin,self.prange)#该行为与最小值的距离
        #整除的意思
        self.zeros_act_grad_filter = tf.zeros([self.action_size])
        self.act_grad = tf.placeholder(tf.float32,[None,self.action_size])
        self.grad_inverter = tf.select(tf.greater(self.act_grad,self.zeros_act_grad_filter),
                                        tf.mul(self.act_grad,self.pdiff_max),tf.mul(self.act_grad,self.pdiff_min))
                                        #感觉应该是梯度比0大的时候，选择梯度乘以与最大值的距离再除以总距离。
                                        #梯度比0小的时候，梯度乘以与最小值的距离再除以总距离
                                        
                                        #这是一种对梯度的滤波，然而原理是什么呢？先加上再说
    def invert(self,grad,action):
        return self.sess.run(self.grad_inverter,feed_dict = {self.action_input:action,self.act_grad:grad[0]})
    