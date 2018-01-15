from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4 #number of hidden layers
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length
def generateData():
    x = np.array(np.random.choice(2, total_series_length, p = [0.5,0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    return (x, y)

batchX = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])#a batchsize by backprop matrix, the windeow that things will happen in
batchY = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])#labels

init_state = tf.placeholder(tf.float32, [batch_size, state_size]) #this is the previous hidden layer

W = tf.Variable(np.random.rand(state_size+1, state_size), dtype =tf.float32) #w and b feed to prev
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype =tf.float32) #these are forward prop
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

input_series = tf.unstack(batchX, axis  =1)
label_series = tf.unstack(batchY, axis  =1)

current_state = init_state
states_series = []
for current_input in input_series:
    current_input = tf.reshape(current_input, [batch_size,1])
    input_and_state_combined = tf.concat(1, [current_input, current_state])