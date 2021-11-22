# %%
import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd

from easydict import EasyDict as edict

# %%
torch.manual_seed(470)
torch.cuda.manual_seed(470)

os.makedirs("images", exist_ok=True)

opt = edict()
opt.n_epochs = 200
opt.batch_size = 64
opt.lr = 0.00005
opt.n_cpu = 4
opt.latent_dim = 100
opt.img_size = 28
opt.channels = 1
opt.clip_value = 0.01
opt.sample_interval = 400
opt.n_critic = 5
opt.b1 = 0.5
opt.b2 = 0.999
print(opt)


# %%
os.makedirs("../images", exist_ok=True)

# %%
img_shape = (opt.channels, opt.img_size, opt.img_size)
img_shape

# %%
cuda = True if torch.cuda.is_available() else False
torch.cuda.is_available()

# %%
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

# %%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


    

# %%
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# %%
dataset = datasets.MNIST("../data/mnist", train=True,
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
                                  download=True)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# %%
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
generator = generator.cuda() if cuda else generator
discriminator = discriminator.cuda() if cuda else discriminator



# %%
# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# %%
batches_done = 0
for epoch in range(opt.n_epochs):
    total_loss_D = 0
    D_num = 0
    total_loss_G = 0
    G_num = 0
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
        total_loss_D += loss_D.item()
        D_num += 1
        loss_D.backward()
        optimizer_D.step()
        
        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))
            total_loss_G += loss_G.item()
            G_num += 1
            loss_G.backward()
            optimizer_G.step()

    

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
        batches_done += 1

    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, opt.n_epochs, total_loss_D/D_num, total_loss_G/G_num)
    )

# %%
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# %%
os.makedirs(f"./log_{0}", exist_ok=True)
log_dir = f"./log_{0}"
writer = SummaryWriter(log_dir)

# %%

# Initialize generator and discriminator
generator_1 = Generator()
discriminator_1 = Discriminator()
generator_1 = generator_1.cuda() if cuda else generator_1
discriminator_1 = discriminator_1.cuda() if cuda else discriminator_1

# Loss weight for gradient penalty
lambda_gp = 10
# Optimizers
optimizer_G = torch.optim.Adam(generator_1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator_1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



batches_done = 0
for epoch in range(opt.n_epochs):
    total_loss_D = 0
    D_num = 0
    total_loss_G = 0
    G_num = 0
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator_1(z)

        # Real images
        real_validity = discriminator_1(real_imgs)
        # Fake images
        fake_validity = discriminator_1(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator_1, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        
        total_loss_D += d_loss.item()
        D_num +=1

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator_1(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator_1(fake_imgs)
            g_loss = -torch.mean(fake_validity)
            total_loss_G += g_loss.item()
            G_num += 1
            
            g_loss.backward()
            optimizer_G.step()

            if batches_done % opt.sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

            batches_done += opt.n_critic

    print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
    % (epoch, opt.n_epochs, total_loss_D/D_num, total_loss_G/G_num))

# %%



