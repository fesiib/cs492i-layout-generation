root = './'
import os
from pathlib import Path

import cv2
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.distributions
from torch.nn.modules import loss


from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_CNN import Discriminator, LayoutEncoder, Generator
from preprocess_CNN import init_dataset
from utils_CNN import process_batch_to_imgs
# from test import test

from utils_CNN import SortByRefSlide, get_device, get_args, get_img_bbs, get_Tensor

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
    z
):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, z)
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
    def inner_save(path, chkp, params):
        torch.save(
        { 
            'epoch' : epoch,
            'model_encoder_state_dict': models['encoder'].state_dict(),
            'model_generator_state_dict': models['generator'].state_dict(),
            'model_discriminator_state_dict': models['discriminator'].state_dict(),
            'optimizer_encoder_state_dict': optimizers['encoder'].state_dict(),
            'optimizer_generator_state_dict': optimizers['generator'].state_dict(),
            'optimizer_discriminator_state_dict': optimizers['discriminator'].state_dict()
        }, os.path.join(path, chkp))
        params_path = os.path.join(path, params)
        with open(params_path, 'w') as f:
            f.write(json.dumps(args, separators=(',\n', ':')))
    
    inner_save(ckpt_dir, f"checkpoint_{epoch}.pt", f"args_{epoch}.json")
    inner_save(ckpt_dir, f"checkpoint_last.pt", f"args_last.json")

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
            clipping = args.enable_clipping,
            L1_loss=args.enable_L1_loss,
            gp = args.enable_gp,
        )
        if (epoch+1) % args.save_period == 0:
            save_checkpoint(models, optimizers, checkpoint_dir, epoch)

def get_l1_loss(fake_layouts, real_layouts):
    return F.l1_loss(fake_layouts, real_layouts)


def run_epoch(
    epoch, models, optimizers,
    is_train=True, dataloader=None, writer = None,
    clipping = False, L1_loss=False, gp=False):
    
    criterion = nn.MSELoss()
    total_loss_G = 0
    total_loss_D = 0
    total_l1_D = 0
    total_gp_D = 0

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

    shape = None
    real_layouts_bbs = None
    fake_layouts_bbs = None
    ref_length = None
    ref_types = None

    for i, batch in enumerate(dataloader):
        batch = SortByRefSlide(batch)
        batch = process_batch_to_imgs(batch, deck=False)
        h, w = batch["shape"][0]
        h, w = int(h), int(w)

        
        # ['shape', 'ref_slide', 'ref_types', 'slide_deck', 'lengths_slide_deck', 'length_ref_types']

        # conditioning
        # shape = batch["shape"].to(device)
        # x_slide_deck = batch["slide_deck"].to(device)
        # length_ref = batch["length_ref_types"].to(device)
        ref_types = batch["ref_types"].to(device)
        ref_slide = batch["ref_slide"].to(device)

        # x_slide_deck = torch.transpose(x_slide_deck, 0, 1)
        # x_slide_deck = torch.transpose(x_slide_deck, 1, 2)
        # lengths_slide_deck = torch.transpose(batch["lengths_slide_deck"], 0, 1).to(device)
        
        optimizers["encoder"].zero_grad()
        optimizers["discriminator"].zero_grad()



        # Sample noise as generator input
        layout_mu, logVar = models['encoder'](ref_slide)
        batch_size, _ = ref_types.shape
        # sample z from q
        std = torch.exp(logVar / 2)
        q = torch.distributions.Normal(layout_mu, std)
        z_hat = q.rsample()
        
        generated_reconstruction = models["generator"](z_hat).detach()
        L_rec = criterion(generated_reconstruction - real_layouts_bbs)
        L_kl =  0.5 * torch.sum(-1 - logVar + layout_mu.pow(2) + logVar.exp())
        L_encoder = L_kl + L_rec
        L_encoder.backward()
        optimizers["encoder"].step()


        optimizers["encoder"].zero_grad()
        layout_mu, logVar = models['encoder'](ref_slide).detach()
        batch_size, _ = ref_types.shape
         # sample z from q
        std = torch.exp(logVar / 2)
        q = torch.distributions.Normal(layout_mu, std)
        z_hat = q.rsample()
        z = torch.autograd.Variable(Tensor(np.random.normal(0, 1, (batch_size, args.layout_encoder_dim))))
        real_layouts_bbs = ref_slide
        fake_layouts_bbs = models['generator'](z).detach()



        # gradient_penalty = compute_gradient_penalty(
        #     models['discriminator'],
        #     real_layouts_bbs,
        #     fake_layouts_bbs,
        #     z
        # )

        loss_D_real = 0.5* criterion(models["discriminator"](real_layouts_bbs, z_hat), torch.ones((batch_size, 1)))
        loss_D_fake = 0.5 * criterion(models["discriminator"](fake_layouts_bbs, z), torch.zeros((batch_size, 1)))
        loss_D = loss_D_real + loss_D_fake
        total_loss_D_fake += loss_D_fake.item()
        total_loss_D_real += loss_D_real.item()
        total_loss_D += loss_D.item()
        D_num += 1
        loss_D.backward()
        optimizers["discriminator"].step()
        

        

        # if L1_loss:
        #     loss_D_l1 = args.lamda_l1 * get_l1_loss(fake_layouts_bbs, real_layouts_bbs)
        #     loss_D += loss_D_l1
        #     total_l1_D += loss_D_l1
        
        # if gp:
        #     loss_D_grad_penalty = args.lambda_gp * gradient_penalty
        #     loss_D += loss_D_grad_penalty
        #     total_gp_D += loss_D_grad_penalty


        

        # Clip weights of discriminator
        if clipping:
            for p in models["discriminator"].parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

        # Train the generator every n_critic iterations
        if i % args.n_critic == 0:
            optimizers["encoder"].zero_grad()
            optimizers["generator"].zero_grad()
            optimizers["discriminator"].zero_grad()

            layout_mu, logVar = models['encoder'](ref_slide)
            batch_size, _ = ref_types.shape
            # sample z from q
            std = torch.exp(logVar / 2)
            q = torch.distributions.Normal(layout_mu, std)
            z_hat = q.rsample()
            
            generated_reconstruction = models["generator"](z_hat)
            L_rec = criterion(generated_reconstruction - ref_slide)
            
            d_losses = []
            # diversity loss
            for i in range(args.gaus_K):
                z_temp = torch.autograd.Variable(Tensor(np.random.normal(0, 1, (batch_size, args.layout_encoder_dim))))
                l = torch.mean((models["generator"](z_temp) - ref_slide).pow(2), dim=(1, 2, 3))
                d_losses.append(l)
            loss_diversity = torch.mean(torch.min(torch.cat(d_losses, dim = 1), dim = 1))



            fake_layouts_bbs = models['generator'](z)
            
            # Adversarial loss
            loss_G = criterion(models["discriminator"](ref_types, fake_layouts_bbs), torch.ones((batch_size, 1)))
            
            loss_G += (loss_diversity + L_rec)
            total_loss_G += loss_G.item()
            
            G_num += 1
            loss_G.backward()
            optimizers["generator"].step()
            optimizers["encoder"].step()
    
    if writer is not None:
        if L1_loss:
            writer.add_scalar('Loss/L1-loss', total_l1_D / D_num, epoch)
        if gp:
            writer.add_scalar('Loss/GP-loss', total_gp_D / D_num, epoch)
        writer.add_scalar('Loss/Discriminator', total_loss_D/D_num, epoch)
        writer.add_scalar('Loss/Generator', total_loss_G/G_num, epoch)
        writer.add_scalar('Loss/Discriminator Real', total_loss_D_real/D_num, epoch)
        writer.add_scalar('Loss/Discriminator Fake', total_loss_D_fake/D_num, epoch)
        if (
            shape is not None and
            real_layouts_bbs is not None and 
            fake_layouts_bbs is not None and
            ref_length is not None and
            ref_types is not None
        ):
            batch_size, _, _ = real_layouts_bbs.shape
            reals = []
            for i in range(args.num_image):
                img = get_img_bbs(shape[i], real_layouts_bbs[i,:ref_length[i],:], ref_types[i,:ref_length[i]], normalized=args.normalized)
                img = cv2.resize(img, (args.image_H, args.image_W))
                img = np.transpose(img, (2, 0, 1))
                reals.append(img)
            reals = np.concatenate(np.expand_dims(reals, axis=0))
            writer.add_images('real layouts', reals, epoch)
            
            fakes = []
            for i in range(args.num_image):
                img = get_img_bbs(shape[i], fake_layouts_bbs[i,:ref_length[i],:], ref_types[i,:ref_length[i]], normalized=args.normalized)
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
    print(device)
    (train_dataset, test_dataset) = init_dataset(False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    encoder = LayoutEncoder().to(device)
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    models = {
        "discriminator": discriminator,
        "encoder" : encoder,
        "generator" : generator,
    }

    optimizers = {
        "discriminator": torch.optim.Adam(models["discriminator"].parameters(), args.lr, args.betas, args.eps),
        "generator": torch.optim.Adam(models["generator"].parameters(), args.lr, args.betas, args.eps),
        "encoder" : torch.optim.Adam(models["encoder"].parameters(), args.lr, args.betas, args.eps)
    }

    num_trial=0
    parent_dir = result_dir / f'trialCNN_{num_trial}'
    while parent_dir.is_dir():
        num_trial = int(parent_dir.name.replace('trialCNN_',''))
        parent_dir = result_dir / f'trialCNN_{num_trial+1}'

    # Modify parent_dir here if you want to resume from a checkpoint, or to rename directory.
    # parent_dir = result_dir / 'trial_8'
    print(f'Logs and ckpts will be saved in : {parent_dir}')

    log_dir = parent_dir
    ckpt_dir = parent_dir
    writer = SummaryWriter(log_dir)

    run_epochs(models, optimizers, train_loader, test_loader, checkpoint_dir=ckpt_dir, writer = writer, load_last = True)

    #test(models, optimizers, test_loader)
    

if __name__ == '__main__':
    train()