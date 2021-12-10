import matplotlib as mpl
import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from easydict import EasyDict as edict
from functools import cmp_to_key
from matplotlib.backends.backend_agg import FigureCanvasAgg

BB_TYPES = [
    '<pad>',
    'title',
    'header',
    'text box',
    'footer',
    'picture',
    'instructor',
    'diagram',
    'table',
    'figure',
    'handwriting',
    'chart',
    'schematic diagram',
]

args = edict()

# General
args.batch_size = 64
args.n_epochs = 10000

args.lr = 0.00005

args.save_period = 50
args.gpu = True

# Data
args.slide_deck_N = 4
args.max_seq_length = 8
args.num_label = len(BB_TYPES)
args.padding_idx = 0

args.image_H = 400
args.image_W = 400
args.num_image = 4

args.train_portion = 0.7
args.normalized = True

# GAN
args.n_cpu = 4
args.latent_vector_dim = 100
args.channels = 1
args.clip_value = 0.1
args.n_critic = 5
args.b1 = 0.5
args.b2 = 0.999


args.slide_deck_embedding_size = 512

# LSTM
args.lamda_l1 = 100
args.lambda_gp = 10

args.enable_L1_loss = True
args.enable_gp = True
args.enable_clipping = False

args.nlayers = 2
args.embedding_size = 2
args.ninp = 4 + args.embedding_size
args.nhid = 512 #512
args.dropout = 0.5

# Transformer
args.enable_reconst = False

args.latent_size = 4

args.G_d_model=256
args.G_nhead=4
args.G_num_layers=8

args.D_d_model=256
args.D_nhead=4
args.D_num_layers=8

device = 'cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu'
Tensor = torch.cuda.FloatTensor if device == 'cuda:0' else torch.FloatTensor

def get_Tensor():
   return Tensor

def get_z():
    z = torch.autograd.Variable(Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_vector_dim))))
    return z

def get_device():
   return device

def get_args():
   return args

def get_bb_types():
   return BB_TYPES

def SortByRefSlide(batch):
    idx = [*range(batch["ref_slide"].shape[0])]

    def by_length(p1, p2):
        return batch["length_ref_types"][p2] - batch["length_ref_types"][p1]
    idx = sorted(idx, key=cmp_to_key(by_length))

    idx = torch.tensor(idx).to(device).long()
    for prop in batch.keys():
        batch[prop] = batch[prop][idx]
    
    return batch

def draw_bbs(shape, bbs, labels, normalized=True):
    if (torch.is_tensor(bbs)):
        bbs = np.array(bbs.tolist())
    if (torch.is_tensor(shape)):
        [h, w] = np.array(shape.tolist())
        shape = (h, w)
    if (torch.is_tensor(labels)):
        labels = np.array(labels.tolist())
    
    eh, ew = shape
    sh, sw = 0, 0
    if normalized:
        sh, sw = -eh, -ew
        eh, ew = eh*2, ew*2

    fig, ax = plt.subplots(1)
    background=patches.Rectangle((sw, sh), ew, eh, linewidth=2, edgecolor='b', facecolor='black')
    ax.add_patch(background)

    ax2 = fig.add_axes([0.92, 0.1, 0.03, 0.8])

    cmap = plt.cm.get_cmap('Set3')
    bounds = np.linspace(1, args.num_label, args.num_label)
    norm = mpl.colors.BoundaryNorm(bounds, args.num_label)
    color_bar = mpl.colorbar.ColorbarBase(
        ax2,
        cmap=cmap,
        spacing='proportional',
        norm=norm,
        ticks=bounds,
        boundaries=bounds,
        format='%1i'
    )
    
    for label, bb in zip(labels, bbs):
        if (label < 1):
            continue
        rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='white', facecolor=cmap(label-1))
        ax.add_patch(rect)
    ax.autoscale(True, 'both')
    plt.show()
    return

def get_img_bbs(shape, bbs, labels, normalized=True):
    if (torch.is_tensor(bbs)):
        bbs = np.array(bbs.tolist())
    if (torch.is_tensor(shape)):
        [h, w] = np.array(shape.tolist())
        shape = (h, w)
    if (torch.is_tensor(labels)):
        labels = np.array(labels.tolist())
    
    eh, ew = shape
    sh, sw = 0, 0
    if normalized:
        sh, sw = -eh, -ew
        eh, ew = eh*2, ew*2

    fig, ax = plt.subplots(1)
    background=patches.Rectangle((sw, sh), ew, eh, linewidth=2, edgecolor='b', facecolor='black')
    ax.add_patch(background)

    ax2 = fig.add_axes([0.92, 0.1, 0.03, 0.8])

    cmap = plt.cm.get_cmap('Set3')
    bounds = np.linspace(1, args.num_label, args.num_label)
    norm = mpl.colors.BoundaryNorm(bounds, args.num_label)
    color_bar = mpl.colorbar.ColorbarBase(
        ax2,
        cmap=cmap,
        spacing='proportional',
        norm=norm,
        ticks=bounds,
        boundaries=bounds,
        format='%1i'
    )
    
    for label, bb in zip(labels, bbs):
        if (label < 1):
            continue
        rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='white', facecolor=cmap(label-1))
        ax.add_patch(rect)
    ax.autoscale(True, 'both')

    canvas = FigureCanvasAgg(fig)
    plt.close(fig)
    
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:,:,:3]


def get_BB_types(bbs):
    return bbs[:, 4]