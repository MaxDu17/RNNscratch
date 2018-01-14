import tensorflow as tf
import numpy as np

LEARNING_RATE =0.1
INPUT_NODES = 2
HIDDEN_NODES = 16
OUTPUT_NODES = 1

int2binary = {}
binary_dim = 8

largest_number = 2 ** binary_dim

for i in range(largest_number):
    int2binary[i] = int(format(i,'b'))

input_layer = tf.Variable(tf.random_normal(shape = [INPUT_NODES,HIDDEN_NODES], stddev = 0.01))
hidden_layer = tf.Variable(tf.random_normal(shape = [HIDDEN_NODES,OUTPUT_NODES], stddev = 0.01))
past_hidden_layer = tf.Variable(tf.random_normal(shape = [HIDDEN_NODES, HIDDEN_NODES], stddev = 0.01))

for j in range (10000):
    a_int = np.random.randint(largest_number/2)
    a = int2binary(a_int)

    b_int = np.random.randint(largest_number / 2)
    b = int2binary(b_int)

    c_int = a_int + b_int
    c = int2binary[c_int]

    best_guess np.zeros_like(c)