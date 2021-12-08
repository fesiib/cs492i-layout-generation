root = './'
import os
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Discriminator, SlideDeckEncoder, Generator
from preprocess import init_dataset
from test import test

from utils import SortByRefSlide, get_device, get_args, get_img_bbs, get_Tensor

# Basic settings
torch.manual_seed(470)
torch.cuda.manual_seed(470)

args = get_args()
device = get_device()

Tensor = get_Tensor()

result_dir = Path(root) / 'results'
result_dir.mkdir(parents=True, exist_ok=True)


def compute_gradient_penalty(
    D,
    real_samples,
    fake_samples,
    ref_types,
    slide_deck_embedding,
    length_ref
):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(ref_types, interpolates, slide_deck_embedding, length_ref)
    fake = torch.autograd.Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)
    gradient_penalty = torch.mean(gradient_penalty)
    return gradient_penalty

def save_checkpoint(models, optimizers, ckpt_dir, epoch):
    path = str(os.path.join(ckpt_dir, f"checkpoint_{epoch}.pt"))
    torch.save(
    { 
        'epoch' : epoch,
        'model_encoder_state_dict': models['encoder'].state_dict(),
        'model_generator_state_dict': models['generator'].state_dict(),
        'model_discriminator_state_dict': models['discriminator'].state_dict(),
        'optimizer_encoder_state_dict': optimizers['encoder'].state_dict(),
        'optimizer_generator_state_dict': optimizers['generator'].state_dict(),
        'optimizer_discriminator_state_dict': optimizers['discriminator'].state_dict()
    }, path)
    path = str(os.path.join(ckpt_dir, f"checkpoint_last.pt"))
    torch.save(
    { 
        'epoch' : epoch,
        'model_encoder_state_dict': models['encoder'].state_dict(),
        'model_generator_state_dict': models['generator'].state_dict(),
        'model_discriminator_state_dict': models['discriminator'].state_dict(),
        'optimizer_encoder_state_dict': optimizers['encoder'].state_dict(),
        'optimizer_generator_state_dict': optimizers['generator'].state_dict(),
        'optimizer_discriminator_state_dict': optimizers['discriminator'].state_dict()
    }, path)

def load_chekpoint(path, models, optimizers):
    path = os.path.join(path, 'checkpoint_last.pt')
    print(path)
    try:
       checkpoint = torch.load(path)
    except:
        print("Couldn't load the last checkpoint!")
        return models, optimizers, -1

    models['encoder'].load_state_dict(checkpoint['model_encoder_state_dict'])
    models['discriminator'].load_state_dict(checkpoint['model_discriminator_state_dict'])
    models['generator'].load_state_dict(checkpoint['model_generator_state_dict'])
    
    optimizers['encoder'].load_state_dict(checkpoint['optimizer_encoder_state_dict'])
    optimizers['generator'].load_state_dict(checkpoint['optimizer_generator_state_dict'])
    optimizers['discriminator'].load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
    epoch = checkpoint['epoch']

    return models, optimizers, epoch


def run_epochs(models, optimizers, train_dataloader, test_dataloader, checkpoint_dir, writer = None, load_last = False):
    loaded_epoch = -1
    if load_last:
        (models, optimizers, loaded_epoch) = load_chekpoint(checkpoint_dir, models, optimizers)
    
    for epoch in range(loaded_epoch + 1, args.n_epochs):
        D_loss, G_loss = run_epoch(
            epoch,
            models,
            optimizers,
            is_train=True,
            dataloader=train_dataloader,
            writer = writer,
            clipping = False,
            L1_loss=args.enable_L1_loss
        )
        if (epoch+1) % args.save_period == 0:
            save_checkpoint(models, optimizers, checkpoint_dir, epoch)

def get_l1_loss(fake_layouts, real_layouts):
    
    return F.l1_loss(fake_layouts, real_layouts)


def run_epoch(epoch, models, optimizers, is_train=True, 
    dataloader=None, writer = None, clipping = False, L1_loss=True):
        
    total_loss_G = 0
    total_loss_D = 0
    G_num = 0
    D_num = 0
    loss_G = 0
    loss_D = 0
    total_loss_D_fake = 0
    total_loss_D_real = 0

    if is_train:
        for model in models:
            models[model].train()
    else:
        for model in models:
            models[model].eval()

    real_layouts_bbs = None
    real_layouts_bbs = None

    for i, batch in enumerate(dataloader):
        batch = SortByRefSlide(batch)

        # ['shape', 'ref_slide', 'ref_types', 'slide_deck', 'lengths_slide_deck', 'length_ref_types']

        # conditioning
        x_slide_deck = batch["slide_deck"].to(device)
        length_ref = batch["length_ref_types"].to(device)
        ref_types = batch["ref_types"].to(device)
        ref_slide = batch["ref_slide"].to(device)

        x_slide_deck = torch.transpose(x_slide_deck, 0, 1)
        x_slide_deck = torch.transpose(x_slide_deck, 1, 2)
        lengths_slide_deck = torch.transpose(batch["lengths_slide_deck"], 0, 1).to(device)
        
        optimizers["encoder"].zero_grad()
        optimizers["discriminator"].zero_grad()

        slide_deck_embedding = models['encoder'](x_slide_deck, lengths_slide_deck)
        
        batch_size, _ = ref_types.shape

        # Sample noise as generator input
        z = torch.autograd.Variable(Tensor(np.random.normal(0, 1, (batch_size, args.latent_vector_dim))))
        
        #   x (tensor): bb labels, (Batch_size, Sequence_size)
        #     z (tensor): latent vector, (Batch_size, latent_vector_dim)
        #     slide_deck_embedding (tensor): slide_deck_embedding vector, (Batch_size, slide_deck_embedding_dim)
        #     length (tensor): (Batch_size,)
        # (batch_size, seq, 4)
        
        # Configure input
        # both have ref_types
        
        real_layouts_bbs = ref_slide[:,:,:-1]

        fake_layouts_bbs = models['generator'](ref_types, z, slide_deck_embedding, length_ref)[0].detach()

        gradient_penalty = compute_gradient_penalty(
            models['discriminator'],
            real_layouts_bbs,
            fake_layouts_bbs,
            ref_types,
            slide_deck_embedding,
            length_ref
        )


        loss_D_real = -torch.mean(models["discriminator"](ref_types, real_layouts_bbs, slide_deck_embedding, length_ref))
        loss_D_fake = torch.mean(models["discriminator"](ref_types, fake_layouts_bbs, slide_deck_embedding, length_ref))
        loss_D_grad_penalty = args.lambda_gp * gradient_penalty
        
        loss_D = loss_D_real + loss_D_fake
        
        if L1_loss:
            loss_D += args.lamda_l1 * get_l1_loss(fake_layouts_bbs, real_layouts_bbs)

        total_loss_D_fake += loss_D_fake.item()
        total_loss_D_real += loss_D_real.item()
        total_loss_D += loss_D.item()
        D_num += 1
        loss_D.backward()
        optimizers["discriminator"].step()
        optimizers["encoder"].step()

        # Clip weights of discriminator
        if clipping:
            for p in models["discriminator"].parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

        # Train the generator every n_critic iterations
        if i % args.n_critic == 0:
            optimizers["encoder"].zero_grad()
            optimizers["generator"].zero_grad()

            slide_deck_embedding = models['encoder'](x_slide_deck, lengths_slide_deck)
            layouts_bbs = models['generator'](ref_types, z, slide_deck_embedding, length_ref)[0]
            
            # Adversarial loss
            loss_G = -torch.mean(models["discriminator"](ref_types, layouts_bbs, slide_deck_embedding, length_ref))
            total_loss_G += loss_G.item()
            G_num += 1
            loss_G.backward()
            optimizers["generator"].step()
            optimizers["encoder"].step()
    
    
    if writer is not None:
        writer.add_scalar('Loss/Discriminator', total_loss_D/D_num, epoch)
        writer.add_scalar('Loss/Generator', total_loss_G/G_num, epoch)
        writer.add_scalar('Loss/Discriminator Real', total_loss_D_real/D_num, epoch)
        writer.add_scalar('Loss/Discriminator Fake', total_loss_D_fake/D_num, epoch)
        if real_layouts_bbs is not None and fake_layouts_bbs is not None:
            batch_size, _, _ = real_layouts_bbs.shape
            reals = []
            for i in range(batch_size//4):
                img = get_img_bbs((1, 1), real_layouts_bbs[i,:,:])
                img = cv2.resize(img, (args.image_H, args.image_W))
                img = np.transpose(img, (2, 0, 1))
                reals.append(img)
            reals = np.concatenate(np.expand_dims(reals, axis=0))
            writer.add_images('real layouts', reals, epoch)
            
            fakes = []
            for i in range(batch_size//4):
                img = get_img_bbs((1, 1), fake_layouts_bbs[i,:,:])
                img = cv2.resize(img, (args.image_H, args.image_W))
                img = np.transpose(img, (2, 0, 1))
                fakes.append(img)
            fakes = np.concatenate(np.expand_dims(fakes, axis=0))
            writer.add_images('fake layouts', fakes, epoch)
    print(
        "[Epoch %d/%d] [D loss: %.4f D real loss: %.4f D fake loss: %.4f| Steps: %d] [G loss: %.4f Steps: %d]"
        % (epoch, args.n_epochs, total_loss_D/D_num, total_loss_D_real/D_num, total_loss_D_fake/D_num, 
        D_num, total_loss_G/G_num, G_num)
    )

    return total_loss_D/D_num, total_loss_G/G_num

def train():
    (train_dataset, test_dataset) = init_dataset()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    encoder = SlideDeckEncoder().to(device)
    discriminator = Discriminator(encoder.slide_encoder.embed.weight.data).to(device)
    generator = Generator(encoder.slide_encoder.embed.weight.data, False).to(device)

    models = {
        "discriminator": discriminator,
        "encoder" : encoder,
        "generator" : generator,
    }

    optimizers = {
        "discriminator": torch.optim.RMSprop(models["discriminator"].parameters(), lr=args.lr),
        "generator": torch.optim.RMSprop(models["generator"].parameters(), lr=args.lr),
        "encoder" : torch.optim.RMSprop(models["encoder"].parameters(), lr=args.lr)
    }
    
    num_trial=0
    parent_dir = result_dir / f'trial_{num_trial}'
    while parent_dir.is_dir():
        num_trial = int(parent_dir.name.replace('trial_',''))
        parent_dir = result_dir / f'trial_{num_trial+1}'

    # Modify parent_dir here if you want to resume from a checkpoint, or to rename directory.
    parent_dir = result_dir / 'trial_2'
    print(f'Logs and ckpts will be saved in : {parent_dir}')

    log_dir = parent_dir
    ckpt_dir = parent_dir
    writer = SummaryWriter(log_dir)

    run_epochs(models, optimizers, train_loader, test_loader, checkpoint_dir=ckpt_dir, writer = writer, load_last = True)

    #test(models, optimizers, test_loader)
    

if __name__ == '__main__':
    train()