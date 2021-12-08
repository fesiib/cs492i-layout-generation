root = './'
import numpy as np
from pathlib import Path

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, SlideDeckEncoder, Generator

from preprocess import init_dataset

from utils import SortByRefSlide, get_device, get_args

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

def run_epoch(models, optimizers, is_train=True, dataloader=None):
    batches_done = 0
    for epoch in range(args.n_epochs):
        total_loss_G = 0
        total_loss_D = 0
        G_num = 0
        D_num = 0
        loss_G = 0
        loss_D = 0
        if is_train:
            for model in models:
                models[model].train()
        else:
            for model in models:
                models[model].eval()
        
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

    #        print("true", real_layouts_bbs[:,:,:4])
   #         print("fake", fake_layouts_bbs[:,:,:4])


  #          print("true: ", models["discriminator"](ref_types, real_layouts_bbs, slide_deck_embedding, length_ref))
 #           print("false: ", models["discriminator"](ref_types, fake_layouts_bbs, slide_deck_embedding, length_ref))

#            break

            loss_D = (-torch.mean(models["discriminator"](ref_types, real_layouts_bbs, slide_deck_embedding, length_ref))
                + torch.mean(models["discriminator"](ref_types, fake_layouts_bbs, slide_deck_embedding, length_ref)))
            #     x (tensor): type labels, (Batch_size, Sequence_size)
            #     bb (tensor): (Batch_size, Sequence_size, 4)
            #     slide_deck_embedding (tensor): slide_deck_embedding vector, (Batch_size, slide_deck_embedding_dim)
            #     length (tensor): (Batch_size,)
            total_loss_D += loss_D.item()
            D_num += 1
            loss_D.backward()
            optimizers["discriminator"].step()
            optimizers["encoder"].step()

            # Clip weights of discriminator
            # for p in models["discriminator"].parameters():
            #     p.data.clamp_(-args.clip_value, args.clip_value)

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
        
            #if batches_done % args.sample_interval == 0:
            batches_done += 1
           
        print(
            "[Epoch %d/%d] [D loss: %f | Steps: %d] [G loss: %f Steps: %d]"
            % (epoch, args.n_epochs, total_loss_D/D_num, D_num, total_loss_G/G_num, G_num)
        )

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