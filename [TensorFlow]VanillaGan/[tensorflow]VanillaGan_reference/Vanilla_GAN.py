# https://arxiv.org/abs/1406.2661
# Generative Adversarial Network(GAN)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# Hyper parameter
total_epoch = 100
batch_size = 100
learning_rate = 0.0001


n_hidden = 256
n_input = 28 * 28

# The amount of noise to use as input to the generator
n_noise = 128

# Since GAN is also an unsupervised learning, it does not use Y like Autoencoder.
X = tf.placeholder(tf.float32, [None, n_input])

# Use noise Z as input value.
Z = tf.placeholder(tf.float32, [None, n_noise])

def generator(noise_z) :
    with tf.variable_scope('generator') :
        hidden = tf.layers.dense(inputs=noise_z, units=n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(inputs=hidden, units=n_input, activation=tf.nn.sigmoid)

    return output

def discriminator(inputs, reuse=None) :
    with tf.variable_scope('discriminator') as scope:
        # In order to make the variables of the models that discriminate the actual image from the images generated by the noise the same,
        # Reuse the previously used variables.

        if reuse :
            scope.reuse_variables()

        hidden = tf.layers.dense(inputs=inputs, units=n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(inputs=hidden, units=1, activation=tf.nn.sigmoid)

    return output

def get_noise(batch_size, n_noise) :
    return np.random.normal(size=(batch_size, n_noise))


# Generate random images using noise
G = generator(Z)

# Returns the value determined using the real image.
D_real = discriminator(X)

# Returns a value that determines whether the image created using noise is a real image.
D_gene = discriminator(G, reuse=True)


"""
According to the paper, optimization of the GAN model maximizes loss_G and loss_D.
We minimize the value of D_gene to maximize loss_D.

This is because...
When you insert the real image in the discriminator, it tries to have the maximum value as: tf.log (D_real) 
And the maximum value as: tf.log (1 - D_gene) even when you insert a fake image.

This makes the discriminator learn the discriminator neural network so that the image produced by the generator is judged to be fake.
"""

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
tf.summary.scalar('loss_D', -loss_D)

"""
On the other hand, to maximize loss_G, we maximize the value of D_gene,
It learns the generator neural network so that when the false image is inserted, the discriminator judges that the image is as real as possible.

In the paper, we find a generator that minimizes to a formula such as loss_D,
This is the same as maximizing the D_gene value, so you can use: loss_G = tf.reduce_mean(tf.log(D_gene))
"""

loss_G = tf.reduce_mean(tf.log(D_gene))
tf.summary.scalar('loss_G', -loss_G)

# If you want to see another loss function, see the following link.
# http://bamos.github.io/2016/08/09/deep-completion/

# When loss_D is obtained, only variables used in the generator neural network are used,
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

# According to the GAN thesis formula, the loss should be maximized, but since the optimization function is used to minimize it, a negative sign is added to loss_D and loss_G to be optimized.
train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-loss_G, var_list=vars_G)

# Start training !
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples/batch_size)
    loss_val_D, loss_val_G = 0, 0

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)

    for epoch in range(total_epoch) :
        for i in range(total_batch) :
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            noise = get_noise(batch_size, n_noise)

            # It learns discriminator and generator neural network separately.
            _, loss_val_D = sess.run([train_D, loss_D],
                                     feed_dict={X : batch_x, Z : noise})
            _, loss_val_G = sess.run([train_G, loss_G],
                                     feed_dict={Z : noise})
        summary = sess.run(merged, feed_dict={X: batch_x, Z: noise})
        writer.add_summary(summary, global_step=epoch)

        print('Epoch:', '%04d' % epoch,
              'D loss: {:.4}'.format(-loss_val_D),
              'G loss: {:.4}'.format(-loss_val_G))

        # Create and save images periodically to see how learning is going
        if epoch == 0 or epoch % 10 == 0 or epoch == total_epoch-1:
            sample_size = 10
            noise = get_noise(sample_size, n_noise)
            samples = sess.run(G,
                               feed_dict={Z : noise})

            fig, ax = plt.subplots(nrows=1, ncols=sample_size, figsize=(sample_size, 1))

            for i in range(sample_size) :
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(samples[i], (28,28)))

            plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

    print('Optimized!')