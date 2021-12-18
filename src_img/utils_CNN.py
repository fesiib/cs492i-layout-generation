from typing import Sequence
import matplotlib as mpl
from numpy.core.fromnumeric import shape
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

NEW_BB_TYPES = [
    '<pad>',
    'header', # -> title, header
    'text box', # -> footer, text box
    'picture', # -> picture
    'figure' # -> instructor, diagram, table, figure, handwriting, chart, schematic diagram
]
bb_encodings = {

    0 : (0, 0, 0), # background
    1 : (0, 0, 1), # -> title, header
    2 : (1, 0, 0),  # -> footer, text box
    3 : (0, 1, 0), # -> picture
    4 : (0, 1, 1) # -> instructor, diagram, table, figure, handwriting, chart, schematic diagram
}

bb_map = { '<pad>' : '<pad>',
        'title' : 'header',
        'header': 'header',
        'text box': 'text box',
        'footer': 'text box',
        'picture': 'picture',
        'instructor': 'figure',
        'diagram': 'figure',
        'table': 'figure',
        'figure': 'figure',
        'handwriting': 'figure',
        'chart': 'figure',
        'schematic diagram': 'figure'
        }

args = edict()

# General
args.batch_size = 64
args.n_epochs = 10000

args.lr = 0.0002
args.betas = (0.5, 0.999)
args.eps = 1e-8

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

# CNN based GAN: LayoutEncoder
args.layout_encoder_dim = 128
args.cond_layout_encoder_dim = 128
args.slide_deck_embedding_output_dim = 128
args.gaus_K = 5

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

def get_new_bb_types():
    return NEW_BB_TYPES

def get_bb_mapping():
    return bb_map

def get_bb_encodings():
    return bb_encodings


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
    ax.invert_yaxis()

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

def indices_array_generic(m,n):
    r0 = np.arange(m) # Or r0,r1 = np.ogrid[:m,:n], out[:,:,0] = r0
    r1 = np.arange(n)
    out = np.empty((m,n,2),dtype=int)
    out[:,:,0] = r0[:,None]
    out[:,:,1] = r1
    return out

def process_batch_to_imgs(batch, deck=True, padded_size=(64,64)):

    batch_size, _ = batch["shape"].shape

    h, w = batch["shape"][0]
    h, w = int(h), int(w)
    print(batch_size, h, w)
    x_slide_deck = batch["slide_deck"]
    length_ref = batch["length_ref_types"]
    ref_types = batch["ref_types"]
    ref_slide = batch["ref_slide"]
    slide_deck_lengths = batch['lengths_slide_deck']
    _, deck_size, sequence_size, _ = x_slide_deck.shape
    slide_img = np.zeros((batch_size, 3, h, w), dtype=np.float32)
    if padded_size is not None:
        slide_img = np.pad(slide_img, ((0, 0), (0, 0), (0, padded_size[0] - h), (0, padded_size[1] - w)))
        mask = indices_array_generic(padded_size[0], padded_size[1])
    else:
        mask = indices_array_generic(h, w)

    enc = get_bb_encodings()

    for key, value in enc.items():
        enc[key] = np.array(value, dtype=np.float32)


    for i in range(batch_size):
        for bb in range(length_ref[i]):
            values = ref_slide[i, bb].tolist()

            _x = int(values[0])
            _y = int(values[1])
            _w = int(values[2])
            _h = int(values[3])
            encoding = enc[int(values[4])]

            
            _mask = (mask[:,:,1] >= _x) &  (mask[:,:,1] < (_w +_x)) &  \
                    (mask[:,:,0] >= _y) &  (mask[:,:,0] <= (_h + _y))

            slide_img[i, :, _mask] = encoding
        # import matplotlib.pyplot as plt
        # import matplotlib.image as img
        # print(slide_img[i].shape)
        # t =slide_img[i]
        # temp = np.transpose(t,(1,2,0))
        # print(temp.shape)
        # plt.imshow(temp)
        # plt.show()
        # print(ref_types[i])
        # print(ref_slide[i])
        # draw_bbs((45, 60), ref_slide[i][:,],ref_types[i], False)
            # x[0] to x[0] + width
            # y[1] to y[1] + height 

    
    slide_img = Tensor(slide_img)
    print(slide_img.shape)
    slide_deck = None

    if deck:
        slide_deck = np.zeros((batch_size, deck_size, 3, h, w), dtype=np.float32)
        if padded_size is not None:
            slide_deck = np.pad(slide_deck, ((0, 0), (0, 0), (0,0), (0, padded_size[0] - h), (0, padded_size[1] - w)))
        for i in range(batch_size):
            for j in range(deck_size):
                for bb in range(slide_deck_lengths[i,j]):

                    values = x_slide_deck[i, j, bb].tolist()
                    _x = int(values[0])
                    _y = int(values[1])
                    _w = int(values[2])
                    _h = int(values[3])
                    encoding =  enc[int(values[4])]
                    
                    _mask = (mask[:,:,1] >= _x) &  (mask[:,:,1] <= (_w +_x)) &  \
                            (mask[:,:,0] >= _y) &  (mask[:,:,0] <= (_h +_y))
                    slide_deck[i, j, :, _mask] = encoding
        slide_deck = Tensor(slide_deck)


    return {
        "shape": batch["shape"],
        "slide_deck": slide_deck,
        "length_ref_types": batch["length_ref_types"],
        "ref_types": batch["ref_types"],
        "ref_slide": slide_img,
        'lengths_slide_deck': batch['lengths_slide_deck']
    }

    



    
