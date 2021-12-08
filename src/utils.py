import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from easydict import EasyDict as edict
from functools import cmp_to_key
from matplotlib.backends.backend_agg import FigureCanvasAgg

args = edict()

args.batch_size = 32
args.nlayers = 2

args.embedding_size = 2
args.ninp = 4 + args.embedding_size
args.nhid = 256 #512

args.dropout = 0.5
args.gpu = True

args.tensorboard = False
args.train_portion = 0.7
args.slide_deck_N = 4
args.slide_deck_embedding_size = 512
args.padding_idx = 0
args.max_seq_length = 8

# Decoder
args.latent_vector_dim = 28

# GAN
args.n_epochs = 20
args.lr = 0.00005
args.n_cpu = 4
args.latent_dim = 100
args.channels = 1
args.clip_value = 0.1
args.sample_interval = 400
args.n_critic = 5
args.b1 = 0.5
args.b2 = 0.999
args.save_period = 20
args.lamda_l1 = 0.3
args.lambda_gp = 10

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

def draw_bbs(shape, bbs):
    if (torch.is_tensor(bbs)):
        bbs = np.array(bbs.tolist())
    if (torch.is_tensor(shape)):
        [h, w] = np.array(shape.tolist())
        shape = (h, w)
    
    h, w = shape
    fig, ax = plt.subplots(1)
    background=patches.Rectangle((-w, -h), w + w, h + h, linewidth=2, edgecolor='b', facecolor='black')
    ax.add_patch(background)
    for bb in bbs:
        rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.autoscale(True, 'both')
    plt.show()
    return

def get_img_bbs(shape, bbs):
    if (torch.is_tensor(bbs)):
        bbs = np.array(bbs.tolist())
    if (torch.is_tensor(shape)):
        [h, w] = np.array(shape.tolist())
        shape = (h, w)
    
    h, w = shape
    fig, ax = plt.subplots(1)
    background=patches.Rectangle((0, 0), w, h, linewidth=2, edgecolor='b', facecolor='black')
    ax.add_patch(background)
    for bb in bbs:
        rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.autoscale(True, 'both')
    canvas = FigureCanvasAgg(fig)
    plt.close(fig)
    
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:,:,:3]


def get_BB_types(bbs):
    return bbs[:, 4]