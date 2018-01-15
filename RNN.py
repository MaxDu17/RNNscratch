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
    int2binary[i] =  [int(x) for x in list('{0:0b}'.format(i))]



input_layer = tf.Variable(tf.random_normal(shape = [INPUT_NODES,HIDDEN_NODES], stddev = 0.01))
hidden_layer = tf.Variable(tf.random_normal(shape = [HIDDEN_NODES,OUTPUT_NODES], stddev = 0.01))
past_hidden_layer = tf.Variable(tf.random_normal(shape = [HIDDEN_NODES, HIDDEN_NODES], stddev = 0.01))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(output_loss)
X = tf.placeholder(tf.int32, [2])
Y = tf.placeholder(tf.int32, [1])
layer_1_values = list()
layer_1_values.append(tf.zeros(HIDDEN_NODES))

hidden_layer_val = tf.sigmoid(np.dot(X, input_layer) + np.dot(layer_1_values[-1], past_hidden_layer))
output = tf.sigmoid(np.dot(hidden_layer_val, hidden_layer))
output_loss = tf.square(Y - output)
with tf.Session() as sess:
    for j in range (10000):
        a_int = np.random.randint(largest_number/2)
        a = int2binary[a_int]

        b_int = np.random.randint(largest_number / 2)
        b = int2binary[b_int]

        c_int = a_int + b_int
        c = int2binary[c_int]

        best_guess = np.zeros_like(c)
        overall_error = 0

        for position in range(binary_dim):
                x = np.array([[a[binary_dim - position -1] , b[binary_dim - position -1]]])
                y = np.array([[c[binary_dim - position - 1]]]).T
        sess.run([optimizer,output_loss], feed_dict = {X: x, Y:y})
