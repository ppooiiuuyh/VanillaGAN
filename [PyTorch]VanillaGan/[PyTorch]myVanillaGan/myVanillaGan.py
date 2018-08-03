import os
import numpy as np
import math
import matplotlib.pyplot as plt


import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

#================================================================
# Hyper parameters [Main]
#================================================================
total_epoch = 100
batch_size = 100
learning_rate = 0.0001
n_hidden = 256
n_input = 28*28
n_noise = 128
cuda = True if torch.cuda.is_available() else False


#================================================================
# dataset load [Utils]
#================================================================
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=batch_size, shuffle=True)



#================================================================
# define modules [Modules]
#================================================================
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.linear_hidden = nn.Linear(input_size, hidden_size)
        self.linear_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear_hidden(x))
        x = F.tanh(self.linear_output(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.linear_hidden = nn.Linear(input_size, hidden_size)
        self.linear_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear_hidden(x))
        x = F.sigmoid(self.linear_output(x))
        return x


class LossDiscriminator(nn.Module):
    def __init__(self):
        super(LossDiscriminator,self).__init__()

    def forward(self, d_real,d_fake):
        loss_d = torch.mean(torch.log(d_real) + torch.log(1-d_fake))
        return loss_d



class LossGenerator(nn.Module):
    def __init__(self):
        super(LossGenerator, self).__init__()

    def forward(self, d_fake):
        loss_g = torch.mean(torch.log(d_fake))
        return loss_g


class NoiseGenerator(nn.Module):
    def __init__(self):
        super(NoiseGenerator,self).__init__()


    def forward(self, batch_size, n_noise):
        return Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], n_noise))))


#=======================================================================
# build models [Models]
#=======================================================================
Tensor = torch.cuda.FloatTensor

generator = Generator(n_noise,n_hidden,n_input).cuda()
discriminator = Discriminator(n_input,n_hidden,1).cuda()
noise_generator = NoiseGenerator()

loss_d = LossDiscriminator()
loss_g = LossGenerator()

optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = torch.optim.Adam([{'params':discriminator.parameters()},
                                {'params':generator.parameters()}],
                               lr=learning_rate)





#=======================================================================
# train [Models]
#=======================================================================

for epoch in range(total_epoch):
# -------------------------------------------------------------
# 1. train discriminator and generator
# -------------------------------------------------------------
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = Variable(Tensor(imgs.reshape(-1,28*28).cuda()))
        z = noise_generator(batch_size,n_noise)
        gen_imgs = generator(z)

        optimizer_d.zero_grad()
        d_loss = -1*loss_d(discriminator(real_imgs),discriminator(gen_imgs.detach()))
        d_loss.backward(retain_graph=True)
        optimizer_d.step()

        optimizer_g.zero_grad()
        g_loss = -1*loss_g(discriminator(gen_imgs))
        g_loss.backward()
        optimizer_g.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
               % (epoch , total_epoch, i, len(dataloader),  d_loss.item(), g_loss.item()))

# -------------------------------------------------------------
# 2. show generated images
# -------------------------------------------------------------
    if epoch % 10 == 0:
    # 2.1 gen fake images
        sample_size = 10
        z = noise_generator(sample_size,n_noise)
        samples = generator(z).detach()

    # 2.2 plot and save generated images
        fig, ax = plt.subplots(nrows=2, ncols=sample_size, figsize=(sample_size, 2))
        for i in range(sample_size):
            ax[0, i].set_axis_off()
            ax[0, i].imshow(np.reshape(samples[i], (28, 28)))

        if not os.path.isdir(os.path.join('./samples')):
            os.makedirs(os.path.join('./samples'), exist_ok=True)
        plt.savefig('./samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
