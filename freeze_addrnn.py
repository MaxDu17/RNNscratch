import os
import sys
import tensorflow as tf
class Hyperparameters:
    INPUT_LAYER = 43
    HIDDEN_LAYER = 100 #Modify??
    OUTPUT_LAYER = 3
    NUM_EPOCHS = 10000
    #NUM_EPOCHS = 1
    BATCH_NUMBER = 240
    LEARNING_RATE = 0.1
    VALIDATION_NUMBER = 30
    TEST_NUMBER = 30
HYP = Hyperparameters()
def create_inference_graph():
    W_Hidd = tf.Variable(tf.random_normal(shape=[HIDDEN + INPUT, HIDDEN], mean=0, stddev=0.1, dtype=tf.float32),
                         name="hidden_weight")  # propagates previous state to current state plus one section for the inputs. Effectively it makes a giant matrix
    B_Hidd = tf.Variable(tf.zeros(shape=[1, HIDDEN]), dtype=tf.float32,
                         name="hidden_bias")  # this bias adds onto the hidden next state

    W_Out = tf.Variable(tf.random_normal(shape=[HIDDEN, OUTPUT], mean=0, stddev=0.1), dtype=tf.float32,
                        name="Out_weight")  # propagates to the end, with output, which is only one thing.
    B_Out = tf.Variable(tf.zeros(shape=[1, 1]), dtype=tf.float32, name="Out_bias")

    states_list = []
    output_list = []

    X = tf.placeholder(tf.float32, shape=[2, binary_dim], name="input")

    iterable_X = tf.unstack(X, axis=1)

    init_hid_layer = tf.placeholder(tf.float32, shape=[1, HIDDEN], name="hidden_layer_state")
    current_hid_layer = init_hid_layer
    first = True
    for current in iterable_X:

        current_flat = tf.reshape(current,[1,INPUT])
        concat_mat = tf.concat([current_flat,current_hid_layer],axis=1)
        next_hid_layer = tf.sigmoid(tf.matmul(concat_mat,W_Hidd) + B_Hidd)

        if first:
            next_states_mat = tf.transpose(next_hid_layer)
            first = False
        else:
            next_states_mat = tf.concat([current_states_mat,tf.transpose(next_hid_layer)],axis = 1)
        current_states_mat = next_states_mat
        current_hid_layer = next_hid_layer
    logit_outputs = tf.matmul(tf.transpose(current_states_mat),W_Out)+B_Out
    prediction_outputs = tf.sigmoid(logit_outputs, name = "output")
create_inference_graph()
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('Best/rough_run-9500.meta', clear_devices=True)
    saver.restore(sess, "Best/rough_run-9500")
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['output']
    )
    tf.train.write_graph(
        frozen_graph_def,
        os.path.dirname("Best/modelv1.pb"),
        os.path.basename("Best/modelv1.pb"),
        as_text=False
    )

