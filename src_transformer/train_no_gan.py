root = './'
import os
from pathlib import Path

import cv2
import json
import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import SlideDeckEncoder
from model import Generator, Discriminator 
from preprocess_root import init_dataset
from utils_transformer import SortByRefSlide, get_device, get_args, get_img_bbs, get_Tensor

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
    bbox_real,
    bbox_fake,
    label,
    deck_enc,
    padding_mask,
):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((bbox_real.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * bbox_real + ((1 - alpha) * bbox_fake)).requires_grad_(True)
    d_interpolates = D(interpolates, label, deck_enc, padding_mask, reconst=False).unsqueeze(-1)
    fake = torch.autograd.Variable(Tensor(bbox_real.shape[0], 1).fill_(1.0), requires_grad=False)
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

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, torch.autograd.Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1).to(device)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return torch.autograd.Variable(y_one_hot) if isinstance(y, torch.autograd.Variable) else y_one_hot

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
    
    #last_eval, best_iou = -1e+8, -1e+8
    for epoch in range(loaded_epoch + 1, args.n_epochs):
        G_loss = run_epoch(
            epoch,
            models,
            optimizers,
            is_train=True,
            dataloader=train_dataloader,
            writer = writer,
        )
        if (epoch+1) % args.save_period == 0:
            save_checkpoint(models, optimizers, checkpoint_dir, epoch)

    # validation
    # fake_layouts = []
    # for model in models:
    #     models[model].eval()
    # with torch.no_grad():
    #     for i, batch in enumerate(test_dataloader):
    #         batch = batch.to(device)
    #         label, mask = to_dense_batch(data.y, data.batch)
    #         bbox_real, _ = to_dense_batch(data.x, data.batch)
    #         padding_mask = ~mask
    #         z = torch.randn(label.size(0), label.size(1),
    #                         args.latent_size, device=device)

    #         bbox_fake = netG(z, label, padding_mask)

    #         fid_val.collect_features(bbox_fake, label, padding_mask)
    #         fid_val.collect_features(bbox_real, label, padding_mask,
    #                                     real=True)

    #         # collect generated layouts
    #         for j in range(label.size(0)):
    #             _mask = mask[j]
    #             b = bbox_fake[j][_mask].cpu().numpy()
    #             l = label[j][_mask].cpu().numpy()
    #             fake_layouts.append((b, l))

    # fid_score_val = fid_val.compute_score()
    # max_iou_val = compute_maximum_iou(val_layouts, fake_layouts)

    # writer.add_scalar('Epoch', epoch, iteration)
    # tag_scalar_dict = {'train': fid_score_train, 'val': fid_score_val}
    # writer.add_scalars('Score/Layout FID', tag_scalar_dict, iteration)
    # writer.add_scalar('Score/Maximum IoU', max_iou_val, iteration)

    # # do checkpointing
    # is_best = best_iou < max_iou_val
    # best_iou = max(max_iou_val, best_iou)

    # save_checkpoint({
    #     'args': vars(args),
    #     'epoch': epoch + 1,
    #     'netG': netG.state_dict(),
    #     'netD': netD.state_dict(),
    #     'best_iou': best_iou,
    #     'optimizerG': optimizerG.state_dict(),
    #     'optimizerD': optimizerD.state_dict(),
    # }, is_best, out_dir)

def run_epoch(
    epoch, models, optimizers,
    is_train=True, dataloader=None, writer = None,
):
        
    total_loss_G = 0

    G_num = 0

    if is_train:
        for model in models:
            models[model].train()
    else:
        for model in models:
            models[model].eval()

    shape = None
    real_layouts_bbs = None
    fake_layouts_bbs = None
    ret_types = None
    ref_length = None

    for i, batch in enumerate(dataloader):
        batch = SortByRefSlide(batch)

        shape = batch["shape"].to(device)
        slide_deck = batch["slide_deck"].to(device)
        lengths_slide_deck = batch["lengths_slide_deck"].to(device)
        ref_length = batch["length_ref_types"].to(device)
        ref_types = batch["ref_types"].to(device).long()
        ref_slide = batch["ref_slide"].to(device)

        # deck encdoing
        slide_deck = torch.transpose(slide_deck, 0, 1)
        lengths_slide_deck = torch.transpose(lengths_slide_deck, 0, 1)
        bboxes = slide_deck[:, :, :, :-1]
        labels = slide_deck[:, :, :, -1].long()
        padding_masks = ~(lengths_slide_deck[:, :, None] > torch.arange(labels.size(2)).to(device)[None, :])

        label = ref_types
        
        padding_mask = ~(ref_length[:, None] > torch.arange(label.size(1)).to(device)[None, :])
        bbox_real = ref_slide[:, :, :-1]
        z = torch.autograd.Variable(Tensor(np.random.normal(0, 1, (label.size(0), label.size(1), args.latent_size))))

        # Update G network
        # D_fake = models['discriminator'](bbox_fake, label, deck_enc, padding_mask)
        # loss_G = F.softplus(-D_fake).mean()
        models['generator'].zero_grad()
        models['encoder'].zero_grad()
        
        deck_enc = models['encoder'](bboxes, labels, padding_masks)

        bbox_fake = models['generator'](z, label, deck_enc, padding_mask)
        #loss_G = F.cross_entropy(logit_cls, packed_label)
        loss_G = F.mse_loss(bbox_fake, bbox_real)
        
        loss_G.backward()
        
        optimizers['generator'].step()
        optimizers['encoder'].step()

        G_num += 1
        total_loss_G += loss_G.item()

        real_layouts_bbs = bbox_real
        fake_layouts_bbs = bbox_fake
    
    if G_num > 0:
        total_loss_G /= G_num

    if writer is not None:
        writer.add_scalar('Loss_transf/Loss_G', total_loss_G, epoch)
        if (
            shape is not None and
            real_layouts_bbs is not None and 
            fake_layouts_bbs is not None and
            ref_length is not None and
            ref_types is not None
        ):
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
        "[Epoch %d/%d] [G loss: %.4f Steps: %d]"
        % (epoch, args.n_epochs, total_loss_G, G_num)
    )

    return total_loss_G

def train():
    print(device)
    (train_dataset, test_dataset) = init_dataset(root, args.normalized)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    encoder = SlideDeckEncoder(
        args.num_label, args.slide_deck_embedding_size, args.slide_deck_N, args.padding_idx, 
        args.D_d_model, args.D_nhead, args.D_num_layers
    ).to(device)

    generator = Generator(
        args.latent_size, args.num_label, args.slide_deck_embedding_size, args.padding_idx,
        d_model=args.G_d_model,
        nhead=args.G_nhead,
        num_layers=args.G_num_layers,
    ).to(device)

    discriminator = Discriminator(
        args.num_label, args.slide_deck_embedding_size, args.max_seq_length, args.padding_idx,
        d_model=args.D_d_model,
        nhead=args.D_nhead,
        num_layers=args.D_num_layers,
    ).to(device)

    models = {
        "discriminator": discriminator,
        "encoder" : encoder,
        "generator" : generator,
    }

    optimizers = {
        "discriminator": torch.optim.Adam(models["discriminator"].parameters(), lr=args.lr),
        "generator": torch.optim.Adam(models["generator"].parameters(), lr=args.lr),
        "encoder" : torch.optim.Adam(models["encoder"].parameters(), lr=args.lr)
    }
    
    num_trial=0
    parent_dir = result_dir / f'trial_transf_{num_trial}'
    while parent_dir.is_dir():
        num_trial = int(parent_dir.name.replace('trial_transf_',''))
        parent_dir = result_dir / f'trial_transf_{num_trial+1}'

    # Modify parent_dir here if you want to resume from a checkpoint, or to rename directory.
    parent_dir = result_dir / 'trial_transf_5'
    print(f'Logs and ckpts will be saved in : {parent_dir}')

    log_dir = parent_dir
    ckpt_dir = parent_dir
    writer = SummaryWriter(log_dir)

    run_epochs(models, optimizers, train_loader, test_loader, checkpoint_dir=ckpt_dir, writer = writer, load_last = True)

    #test(models, optimizers, test_loader)

if __name__ == "__main__":
    train()