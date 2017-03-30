#! /usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

f = open('episode_reward.txt')
lines = f.readlines()
reward = []
for line in lines:
    if float(line)>-300:
        reward.append(float(line))

plt.plot(reward)
plt.show()

f.close()