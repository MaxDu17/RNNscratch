import tensorflow as tf
import numpy as np

epochs = 1000
learning_rate = 0.1

INPUT = 2
HIDDEN = 16
OUTPUT = 1

int2binary = {}
binary_dim = 8

largest_number = 2 ** binary_dim

for i in range(largest_number):
    bin_carr = [int(x) for x in list('{0:0b}'.format(i))]
    length = len(bin_carr)
    zeros = [0]*(8-length)
    int2binary[i] = zeros + bin_carr

W_Hidd = tf.Variable(tf.random_normal(shape = [HIDDEN+INPUT,HIDDEN], stddev = 0.01, dtype =tf.float32))#propagates previous state to current state plus one section for the inputs. Effectively it makes a giant matrix
B_Hidd = tf.Variable(tf.zeros(shape=[1,HIDDEN]), dtype =tf.float32) #this bias adds onto the hidden next state

W_Out = tf.Variable(tf.random_normal(shape = [HIDDEN, OUTPUT], stddev = 0.01),dtype =tf.float32) #propagates to the end, with output, which is only one thing.
B_Out = tf.Variable(tf.zeros(shape = [1,1]), dtype =tf.float32)

states_list = []
output_list = []
X = tf.placeholder(tf.float32, shape=[2,binary_dim])
Y = tf.placeholder(tf.int32, shape = [1,binary_dim])
iterable_X = tf.unstack(X,axis=1)
iterable_Y= tf.unstack(Y,axis=1)
init_hid_layer = tf.placeholder(tf.float32, shape = [1,HIDDEN])
current_hid_layer = init_hid_layer
for current in iterable_X:
    current_flat = tf.reshape(current,[1,INPUT])
    concat_mat = tf.concat([current_flat,current_hid_layer],axis=1)
    curr_hid_layer = tf.matmul(concat_mat,W_Hidd)
    curr_hid_layer = tf.add(curr_hid_layer,B_Hidd)
    states_list.append(curr_hid_layer)

logit_outputs = (tf.matmul(curr_hidd,W_Out)+ B_Out for curr_hidd in states_list)
softmax_pred = (tf.nn.softmax(outputs) for outputs in logit_outputs)
loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels) for logits, labels in zip(logit_outputs, iterable_Y)]
total_loss = tf.reduce_mean(loss)
training = train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(epochs):
        a_int = np.random.randint(largest_number/2)
        a = int2binary[a_int]
        a_np = np.matrix(a)
        #print(a_np)
        b_int = np.random.randint(largest_number / 2)
        b = int2binary[b_int]
        b_np = np.matrix(b)
        c_int = a_int + b_int
        c = int2binary[c_int]
        c_np = np.matrix(c)
        print(c_np)
        x = np.concatenate((a_np,b_np), axis=0)


       # print(x)

        total_loss, _,predictions = sess.run([total_loss,training,softmax_pred],
                                             feed_dict= {X:x,Y:c_np})
        print(total_loss)
        print("ll")
        print(predictions)




