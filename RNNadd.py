import tensorflow as tf
import numpy as np

epochs = 100000
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

W_Hidd = tf.Variable(tf.random_normal(shape = [HIDDEN+INPUT,HIDDEN], mean = 0, stddev = 0.1, dtype =tf.float32), name="hidden_weight")#propagates previous state to current state plus one section for the inputs. Effectively it makes a giant matrix
B_Hidd = tf.Variable(tf.zeros(shape=[1,HIDDEN]), dtype =tf.float32,name="hidden_bias") #this bias adds onto the hidden next state

W_Out = tf.Variable(tf.random_normal(shape = [HIDDEN, OUTPUT], mean = 0 ,stddev = 0.1),dtype =tf.float32,name="Out_weight") #propagates to the end, with output, which is only one thing.
B_Out = tf.Variable(tf.zeros(shape = [1,1]), dtype =tf.float32, name="Out_bias")

states_list = []
output_list = []

X = tf.placeholder(tf.float32, shape=[2,binary_dim],name="input")
Y = tf.placeholder(tf.float32, shape = [1,binary_dim],name = "label")
iterable_X = tf.unstack(X,axis=1)
iterable_Y= tf.unstack(Y,axis=1)
init_hid_layer = tf.placeholder(tf.float32, shape = [1,HIDDEN],name="hidden_layer_state")
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



logit_outputs = tf.add(tf.matmul(tf.transpose(current_states_mat),W_Out),B_Out,name = "prediction_outputs")
#prediction_outputs = tf.sigmoid(logit_outputs,name = "prediction_outputs")
loss = tf.square(tf.subtract(Y,tf.transpose(logit_outputs)))
total_loss = tf.reduce_mean(loss)

training = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

tf.summary.histogram("W_Hidd", W_Hidd)
tf.summary.histogram("W_Out", W_Out)
tf.summary.scalar("Loss", total_loss)
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("MYRNNGRAPH/", sess.graph)
    tf.train.write_graph(sess.graph_def, '.', 'models/bigmodel.pbtxt')
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

        pseudo_curr = np.zeros((1,HIDDEN))
        x = np.concatenate((a_np,b_np), axis=0)
        x = np.flip(x, axis = 1)
        c_np = np.flip(c_np, axis = 1)

        cur_mat,prediction, _total_loss, _,summary = sess.run([current_states_mat,logit_outputs, total_loss,training,summary_op],feed_dict= {X:x,Y:c_np,init_hid_layer:pseudo_curr})
        writer.add_summary(summary, global_step=epoch)
        if epoch %5000 ==0:
            print("batch", epoch)
            print("total loss",_total_loss)
            prediction_rounded = np.abs(np.round(prediction))
            print("raw prediction",np.reshape(prediction,[1,8]))
            print("rounded prediction", np.reshape(prediction_rounded,[1,8]))
            print("actual answer",c_np)
            print("---------------")
            saver.save(sess, "models/test", global_step=epoch)
    writer.close()




