import tensorflow as tf
import numpy as np

epochs = 100
learning_rate = 0.1

INPUT = 2
HIDDEN = 16
OUTPUT = 1

int2binary = {}
binary_dim = 8

largest_number = 2 ** binary_dim

for i in range(largest_number):
    int2binary[i] =  [int(x) for x in list('{0:0b}'.format(i))]

W_Hidd = tf.Variable(tf.random_normal(shape = [HIDDEN+INPUT,HIDDEN], stddev = 0.01, dtype =tf.float32))#propagates previous state to current state plus one section for the inputs. Effectively it makes a giant matrix
B_Hidd = tf.Variable(tf.zeros(shape=[1,HIDDEN]), dtype =tf.float32) #this bias adds onto the hidden next state

W_Out = tf.Variable(tf.random_normal(shape = [HIDDEN, OUTPUT], stddev = 0.01),dtype =tf.float32) #propagates to the end, with output, which is only one thing.
B_Out = tf.Variable(tf.zeros(shape = [1,1]), dtype =tf.float32)

states_list = {}
output_list = []


