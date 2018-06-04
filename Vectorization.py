# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 06:41:53 2018

@author: am21381
"""

import numpy as np



# Run the block multiple times
import time 

a=np.random.rand(1000000)
b=np.random.rand(1000000)

start = time.time()
result = np.dot(a,b)
stop = time.time()
print (result)
print ("Vectorized Version :" +str(1000*(stop-start))+"ms")

result = 0
start = time.time()
for i in range(1000000):
    result = result + a[i]*b[i]
stop = time.time()
print (result)
print ("For Loop Version :" +str(1000*(stop-start))+"ms")

# A code that takes 6 hours in loops would take 1 hour in vectors

# GPU and CPU Paralleizatiion enabled by SIMD (Single instruction multiple data) if we use Python libraries rather than loops

# Whenever possible avoid for loops 
import math
a = np.random.rand(1000,1)
result = np.zeros(shape = (1000,1))
sum = 0
for i in range(1000):
    result[i] = math.exp(a[i]) 
result_2 = np.exp(a)

#Vectozied Operations
abs_value = np.abs(a)
max = np.max(a)

# Whenever tempted to apply for loop , please check if we can do it wihout for loop


    
        