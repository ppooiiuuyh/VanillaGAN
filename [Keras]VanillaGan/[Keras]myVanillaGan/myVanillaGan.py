from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys, os
import numpy as np

#================================================================
# Hyper parameters [Main]
#================================================================
total_epoch = 100
batch_size = 100
learning_rate = 0.0001
n_hidden = 256
n_input = 28 * 28
n_noise = 128

#================================================================
# dataset load [Utils]
#================================================================
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/",one_hot=True)




#================================================================
# define modules [Modules]
#================================================================
def build_generator():
    model = Sequential()
    model.add(Dense(n_hidden, input_dim=n_noise, activation=None))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(n_input, activation='tanh'))
    model.summary()

    noise = Input(shape=(n_noise,))
    img = model(noise)

    return Model(noise, img)


def build_discriminator():
    model = Sequential()
    model.add(Dense(input_dim=n_input, output_dim=n_hidden, activation=None))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=(n_input,))
    validity = model(img)

    return Model(img, validity)

def gen_noise(batch_size, n_noise) :
    return np.random.normal(size=[batch_size,n_noise])




#=======================================================================
# build models [Models]
#=======================================================================
optimizer = Adam(0.0002, 0.5)


discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


discriminator.trainable = False
z = Input(shape=(n_noise,))
generator = build_generator()
gen_img = generator(z)
d_fake = discriminator(gen_img)
model_d_fake = Model(z, d_fake)
model_d_fake.compile(loss='binary_crossentropy', optimizer=optimizer)


#=======================================================================
# train [Models]
#=======================================================================
for epoch in range(total_epoch):
# -------------------------------------------------------------
# 1 train discriminator and generator
# -------------------------------------------------------------
    for i in range(batch_size):
        imgs, batch_y = mnist.train.next_batch(batch_size)

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        noise = gen_noise(batch_size, n_noise)
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = np.add(d_loss_real, d_loss_fake)/2


        g_loss = model_d_fake.train_on_batch(noise, valid)

        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
               % (epoch, d_loss[0], 100*d_loss[1], g_loss))


# -------------------------------------------------------------
# 2. show generated images
# -------------------------------------------------------------
    if epoch % 10 == 0:
    # 3.2.1 gen fake images
        sample_size = 10
        noise = gen_noise(sample_size,n_noise)
        samples = generator.predict(noise)

    # 3.2.2 plot and save generated images
        fig, ax = plt.subplots(nrows=2, ncols=sample_size, figsize=(sample_size, 2))
        for i in range(sample_size):
            ax[0, i].set_axis_off()
            ax[0, i].imshow(np.reshape(samples[i], (28, 28)))

        if not os.path.isdir(os.path.join('./samples')):
            os.makedirs(os.path.join('./samples'), exist_ok=True)
        plt.savefig('./samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)




