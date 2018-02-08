import tensorflow as tf
import numpy as np

epochs = 1000
learning_rate = 0.1

INPUT = 2
HIDDEN = 4
OUTPUT = 1

int2binary = {}
binary_dim = 8

largest_number = 2 ** binary_dim
total_loss =0
for i in range(largest_number):
    bin_carr = [int(x) for x in list('{0:0b}'.format(i))]
    length = len(bin_carr)
    zeros = [0]*(8-length)
    int2binary[i] = zeros + bin_carr
W_In = tf.Variable(tf.random_normal(shape = [INPUT,HIDDEN],mean=0.5,stddev = 0.1, dtype =tf.float32), name="hidden_weight")#propagates previous state to current state plus one section for the inputs. Effectively it makes a giant matrix
#B_In = tf.Variable(tf.zeros(shape=[1,HIDDEN]), dtype =tf.float32, name="hidden_bias") #this bias adds onto the hidden next state

W_Hidd = tf.Variable(tf.random_normal(shape = [HIDDEN,HIDDEN],mean=0.5, stddev = 0.1, dtype =tf.float32), name="hidden_weight")#propagates previous state to current state plus one section for the inputs. Effectively it makes a giant matrix
#B_Hidd = tf.Variable(tf.zeros(shape=[1,HIDDEN]), dtype =tf.float32, name="hidden_bias") #this bias adds onto the hidden next state

W_Out = tf.Variable(tf.random_normal(shape = [HIDDEN, OUTPUT],mean=0.5, stddev = 0.1),dtype =tf.float32,name="Out_weight") #propagates to the end, with output, which is only one thing.
#B_Out = tf.Variable(tf.zeros(shape = [1,1]), dtype =tf.float32, name="Out_bias")


X = tf.placeholder(tf.float32, shape=[1,INPUT],name="input")
Y = tf.placeholder(tf.float32, shape = [1,1],name = "label")


prev_hidd_layer = tf.placeholder(tf.float32, shape = [1,HIDDEN],name="hidden_layer_state")


current_hidd_layer = tf.add(tf.matmul(X,W_In),tf.matmul(prev_hidd_layer, W_Hidd))
current_hidd_layer = tf.sigmoid(current_hidd_layer)
logit_output = tf.matmul(current_hidd_layer,W_Out)
loss = (logit_output-Y)**2
#loss = tf.square(tf.subtract(logit_output,Y))
total_loss = total_loss + loss
training = tf.train.AdagradOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #writer = tf.summary.FileWriter("test_add/", sess.graph)
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

        _current_hidd_layer = np.zeros([1,HIDDEN])

        x = np.concatenate((np.transpose(a_np),np.transpose(b_np)), axis=1)
        output_concat = []
        for position in range(binary_dim):
            frame_in = x
            #print(frame_in[binary_dim-position-1])
            label = np.transpose(c_np)
            #print(label[binary_dim-position-1])
           # print("buffer")
            _current_hidd_layer, _output, _loss,_total_loss,_= sess.run([current_hidd_layer,logit_output,loss,total_loss,training],feed_dict= {
                X:frame_in[binary_dim-position-1],
                Y:label[binary_dim-position-1],
                prev_hidd_layer: _current_hidd_layer
            })
            output_concat.append(_output)
            print(_loss)
        '''if epoch%10 == 0:

            print("buffer")
            print(c_np)
            print(np.reshape(output_concat,[1,8]))
            print(_total_loss)


'''

