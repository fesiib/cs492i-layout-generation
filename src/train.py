root = './'
import numpy as np
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, SlideDeckEncoder, Generator

from preprocess import init_dataset

from utils import SortByRefSlide, get_device, get_args, get_img_bbs

# Basic settings
torch.manual_seed(470)
torch.cuda.manual_seed(470)

args = get_args()
device = get_device()

Tensor = torch.cuda.FloatTensor if device == 'cuda:0' else torch.FloatTensor

result_dir = Path(root) / 'results'
result_dir.mkdir(parents=True, exist_ok=True)

# def save_model(model, epoch="last"):
#     torch.save(model.state_dict(),  result_dir / f'{type(model).__name__}_{mode}.ckpt')

# def load_model(model, epoch="last"):
#     if os.path.exists(result_dir / f'{type(model).__name__}_{mode}.ckpt'):
#         model.load_state_dict(torch.load(result_dir / f'{type(model).__name__}_{mode}.ckpt'))

# def load_model(model, epoch="last"):
#     if os.path.exists(result_dir / f'{type(model).__name__}_{mode}.ckpt'):
#         model.load_state_dict(torch.load(result_dir / f'{type(model).__name__}_{mode}.ckpt'))

# num_trial=0
# parent_dir = result_dir / f'trial_{num_trial}'
# while parent_dir.is_dir():
#     num_trial = int(parent_dir.name.replace('trial_',''))
#     parent_dir = result_dir / f'trial_{num_trial+1}'

# Modify parent_dir here if you want to resume from a checkpoint, or to rename directory.
# parent_dir = result_dir / 'trial_99'
# print(f'Logs and ckpts will be saved in : {parent_dir}')

# log_dir = parent_dir
# ckpt_dir = parent_dir
# encoder_ckpt_path = parent_dir / 'encoder.pt'
# generator_ckpt_path = parent_dir / 'generator.pt'
# discriminator_ckpt_path = parent_dir / 'discriminator.pt'
# writer = SummaryWriter(log_dir)

# def compute_gradient_penalty(D, real_samples, fake_samples):
#     """Calculates the gradient penalty loss for WGAN GP"""
#     # Random weight term for interpolation between real and fake samples
#     alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
#     # Get random interpolation between real and fake samples
#     interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
#     d_interpolates = D(interpolates)
#     fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
#     # Get gradient w.r.t. interpolates
#     gradients = autograd.grad(
#         outputs=d_interpolates,
#         inputs=interpolates,
#         grad_outputs=fake,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True,
#     )[0]
#     gradients = gradients.view(gradients.size(0), -1)
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#     return gradient_penalty

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

def load_chekpoint(path, models, optimizers):
    # path = str(os.path.join(ckpt_dir, f"checkpoint_{epoch}.pt"))
    checkpoint = torch.load(path)
    

    models['encoder'].load_state_dict(checkpoint['model_encoder_state_dict'])
    models['discriminator'].load_state_dict(checkpoint['model_discriminator_state_dict'])
    models['generator'].load_state_dict(checkpoint['model_generator_state_dict'])
    
    optimizers['encoder'].load_state_dict(checkpoint['optimizer_encoder_state_dict'])
    optimizers['generator'].load_state_dict(checkpoint['optimizer_generator_state_dict'])
    optimizers['discriminator'].load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
    epoch = checkpoint['epoch']

    return models, optimizers, epoch


def run_epochs(models, optimizers, train_dataloader, test_dataloader, writer = None, *args, **kwargs):
    


    for epoch in range(args.n_epochs):

        D_loss, G_loss = run_epoch(epoch, models, optimizers, is_train=True, dataloader=train_dataloader, writer = writer)


        
        
        if epoch % args.save_period:

            save_checkpoint(models, optimizers, kwargs['checkpoint_dir'], epoch)
            
            


        # run_epoch(epoch, models, optimizers, is_train=False, dataloader=test_dataloader, writer = writer)

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



        # print("true: ", models["discriminator"](ref_types, real_layouts_bbs, slide_deck_embedding, length_ref))
        # print("false: ", models["discriminator"](ref_types, fake_layouts_bbs, slide_deck_embedding, length_ref))


        loss_D_real = -torch.mean(models["discriminator"](ref_types, real_layouts_bbs, slide_deck_embedding, length_ref))
        loss_D_fake = torch.mean(models["discriminator"](ref_types, fake_layouts_bbs, slide_deck_embedding, length_ref))
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
                img = img.transpose(0, 2)
                img = img.transpose(0, 1)
                reals.append(img)
            reals = np.concatenate(reals)
            writer.add_images('real layouts', reals, epoch)
            
            fakes = []
            for i in range(batch_size//4):
                img = get_img_bbs((1, 1), fake_layouts_bbs[i,:,:])
                img = img.transpose(0, 2)
                img = img.transpose(1, 2)
                reals.append(img)
            fakes = np.concatenate(fakes)
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

    run_epoch(models, optimizers, True, train_loader)
    

if __name__ == '__main__':
    train()