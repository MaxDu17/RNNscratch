from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4 #number of hidden PROPERTIES
num_classes = 2
echo_step = 3
batch_size = 5
learning_rate = 0.3
num_batches = total_series_length//batch_size//truncated_backprop_length
def generateData():
    x = np.array(np.random.choice(2, total_series_length, p = [0.5,0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    return (x, y)

batchXPl = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])#a batchsize by backprop matrix, the windeow that things will happen in
batchYPl = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])#labels

init_state = tf.placeholder(tf.float32, [batch_size, state_size]) #this is the previous hidden layer

W = tf.Variable(np.random.rand(state_size+1, state_size), dtype =tf.float32) #these are for states AND THE INPUTS (note the +1)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype =tf.float32) #these are for secondary propagation
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

input_series = tf.unstack(batchXPl, axis  =1)
label_series = tf.unstack(batchYPl, axis  =1)

current_state = init_state
states_series = []
for current_input in input_series:
    current_input = tf.reshape(current_input, [batch_size,1])
    input_and_state_combined = tf.concat([current_input, current_state],1)
    next_state = tf.tanh(tf.matmul(input_and_state_combined, W) + b)#forward prop everything
    states_series.append(next_state)#add this to list
    current_state = next_state#set current for next state, for next run through

logits_series = [tf.matmul(state,W2) + b2 for state in states_series]#this props everything to the final
#this might be tricky but the first step propagates to state, and not to verdict.
#the current step propagates the states to their verdicts. It's a two step process
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels) for logits, labels in zip(logits_series, label_series)]
#above calculates the loss
total_loss = tf.reduce_mean(losses)
tf.summary.scalar("loss", total_loss)
summary_op = tf.summary.merge_all()
#takes the mean of the loss
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []
    writer = tf.summary.FileWriter("test/", sess.graph)
    for epoch in range(epochs):
        x, y = generateData()
        _current_state = np.zeros((batch_size,state_size))
        print("new epoch", epoch)
        for batch in range(num_batches):
            start_place = batch * truncated_backprop_length #this moves the window forwards
            end_place = start_place + truncated_backprop_length #this sets the end of window

            batchX = x[:,start_place:end_place] #creates batches in window
            batchY = y[:, start_place:end_place] #creates answers array

            _total_loss, _train_step, _current_state, _predictions_series, summary = sess.run(
                [total_loss, train_step, current_state, predictions_series, summary_op],
                feed_dict={
                    batchXPl: batchX,
                    batchYPl: batchY,
                    init_state:_current_state
                }
            )
            loss_list.append(_total_loss)
            writer.add_summary(summary, global_step = epoch)
            if batch%100 == 0:
                print("step", batch, "loss", _total_loss)
        writer.close()
