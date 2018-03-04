import tensorflow as tf
import os
import numpy as np
from make_sets import Setmaker as SM

set_maker = SM()
pbfilename = "models/frozen_modelv1.pb"
int2binary = {}
binary_dim = 8
epochs = 100000
learning_rate = 0.1

INPUT = 2
HIDDEN = 16
OUTPUT = 1

largest_number = 2 ** binary_dim

for i in range(largest_number):
    bin_carr = [int(x) for x in list('{0:0b}'.format(i))]
    length = len(bin_carr)
    zeros = [0]*(8-length)
    int2binary[i] = zeros + bin_carr
with tf.gfile.GFile(pbfilename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def,
                        input_map = None,
                        return_elements = None,
                        name = "")
input = graph.get_tensor_by_name("input:0")
output = graph.get_tensor_by_name("prediction_outputs:0")
init_hid_layer = graph.get_tensor_by_name("hidden_layer_state:0")

with tf.Session(graph=graph) as sess:


    a_int = np.random.randint(largest_number / 2)
    a = int2binary[55]
    a_np = np.matrix(a)
    # print(a_np)
    b_int = np.random.randint(largest_number / 2)
    b = int2binary[99]
    b_np = np.matrix(b)
    c_int = 55+99
    c = int2binary[c_int]
    c_np = np.matrix(c)

    pseudo_curr = np.zeros((1, HIDDEN))
    x = np.concatenate((a_np, b_np), axis=0)
    x = np.flip(x, axis=1)
    c_np = np.flip(c_np, axis=1)

    prediction = sess.run(
        output,
        feed_dict={input: x, init_hid_layer: pseudo_curr})
    print(c_np)
    print(prediction)