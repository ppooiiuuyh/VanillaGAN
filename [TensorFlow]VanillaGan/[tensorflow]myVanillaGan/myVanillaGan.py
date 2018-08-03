import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


#================================================================
# Hyper parameters [Main]
#================================================================
total_epoch = 100
batch_size = 100
learning_rate = 0.0001
n_hidden = 256
n_input = 28 * 28
n_noise = 128

x = tf.placeholder(tf.float32, [None, n_input])
z = tf.placeholder(tf.float32, [None, n_noise])



#================================================================
# dataset load [Utils]
#================================================================
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/",one_hot=True)




#================================================================
# define modules [Modules]
#================================================================
def generator(noise_z) :
    with tf.variable_scope('generator') :
        hidden = tf.layers.dense(inputs=noise_z,units=n_hidden,activation=tf.nn.relu)
        output = tf.layers.dense(inputs=hidden,units =n_input, activation=tf.nn.sigmoid)
    return output,hidden_see


def discriminator(inputs, reuse=None) :
    with tf.variable_scope('discriminator') as scope :
        if reuse : scope.reuse_variables()
        hidden = tf.layers.dense(inputs=inputs, units=n_hidden,activation=tf.nn.relu)
        output = tf.layers.dense(inputs=hidden, units=1, activation=tf.nn.sigmoid)
    return output,hidden_see


def gen_noise(batch_size, n_noise) :
    return np.random.normal(size=[batch_size,n_noise])


#=======================================================================
# build models [Models]
#=======================================================================
g_recons,_ = generator(z) #z = tf.placeholder(tf.float32, [None, n_noise])


d_real,hidden_see = discriminator(x)
d_fake,_ = discriminator(g_recons,reuse = True)

loss_d = tf.reduce_mean(tf.log(d_real) + tf.log(1-d_fake))
loss_g = tf.reduce_mean(tf.log(d_fake))


vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')
vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')
train_d = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-1*loss_d, var_list=vars_d)
train_g = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-1*loss_g, var_list=vars_g)



#=======================================================================
# train [Models]
#=======================================================================
with tf.Session() as sess :
#----------------------------------------------------------------
# 1. init session
#----------------------------------------------------------------
    sess.run(tf.global_variables_initializer())


#-----------------------------------------------------------------
# 2. prepare train
#------------------------------------------------------------------
    total_batch = mnist.train.num_examples//batch_size

#----------------------------------------------------------------
# 3. start train
#------------------------------------------------------------------
    for epoch in range(total_epoch) :
    #-------------------------------------------------------------
    # 3.1 train discriminator and generator
    #-------------------------------------------------------------
        for i in range(total_batch) :
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            noise = gen_noise(batch_size, n_noise)
            loss_val_d,_ = sess.run([loss_d,train_d],feed_dict={x : batch_x, z: noise})
            loss_val_g,_ = sess.run([loss_g,train_g],feed_dict={z : noise})
            print('Epoch:', epoch,'D loss:', -loss_val_d,'G loss:', -loss_val_g)

    #------------------------------------------------------------
    # 3.2 show generated images
    #-------------------------------------------------------------
        if epoch == 0 or epoch % 10 == 0 or epoch == total_epoch-1:
        # 3.2.1 gen fake images
            sample_size = 10
            samples = sess. run(g_recons, feed_dict={z:gen_noise(sample_size, n_noise)})

        # 3.2.2 plot and save generated images
            fig, ax = plt.subplots(nrows=2, ncols=sample_size, figsize=(sample_size, 2))
            for i in range(sample_size):
                ax[0,i].set_axis_off()
                ax[0,i].imshow(np.reshape(samples[i], (28, 28)))


            if not os.path.isdir(os.path.join('./samples')):
                os.makedirs(os.path.join('./samples'), exist_ok=True)
            plt.savefig('./samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)