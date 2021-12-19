root = '../'

import os
from skimage import io, transform
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import mask_rcnn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

# Basic settings
from easydict import EasyDict as edict

cfg = []

torch.manual_seed(470)
torch.cuda.manual_seed(470)

args = edict()
args.batch_size = 1
args.lr = 1e-4
args.momentum = 0.9
args.weight_decay = 5e-4
args.epoch = 10
args.tensorboard = False
args.gpu = True
args.train_portion = 0.7

device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

# Create directory name.
result_dir = Path(root) / 'results'
result_dir.mkdir(parents=True, exist_ok=True)

if args.tensorboard:
    %load_ext tensorboard
    %tensorboard --logdir "{str(result_dir)}" --samples_per_plugin images=100

def draw_image(image, bbs):
    if (torch.is_tensor(image)):
        image = np.array(image.tolist()).transpose((1, 2, 0))
    if (torch.is_tensor(bbs)):
        bbs = np.array(bbs.tolist())

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for bb in bbs:
        rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

class FitVidDataset(Dataset):
    """ FitVid Dataset"""
    
    def __init__(self, img_data, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.img_data = img_data
        self.img_filenames = list(self.img_data.keys())
    
    def __len__(self):
        return len(self.img_filenames)

    def show_image(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.img_filenames[idx]
        img_dir = os.path.join(self.root_dir, filename)
        
        image = io.imread(img_dir)
        draw_image(image, self.img_data[filename]['bbs'])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.img_filenames[idx]
        img_dir = os.path.join(self.root_dir, filename)
        image = io.imread(img_dir)

        bbs = self.img_data[filename]['bbs']

        sample = {
            'image' : image,
            'labels' : bbs,
        }
        
        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        labels = labels * np.array([new_w / w, new_h / h, new_w / w, new_h / h, 1]).T
        return {'image': img, 'labels': labels}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        labels = labels - np.array([left, top, 0, 0, 0]).T

        return {'image': image, 'labels': labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

BB_TYPES = [
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

def process_fitvid_dataset(all_dataset, root_dir):
    img_data = {}
    for entrance in all_dataset.iloc:
        filename = entrance['filename']
        img_dir = os.path.join(root_dir, filename)
        if (os.path.exists(img_dir) is False):
            continue
        if (filename not in img_data):
            img_data[filename] = {
                'bbs': [],
                'img_w' : entrance['Image Width'],
                'img_h' : entrance['Image Height'],
            }
        
        bb_type = BB_TYPES.index(entrance['Type'])
        if (bb_type < 0 or bb_type >= len(BB_TYPES)):
            bb_type = len(BB_TYPES)

        bb = np.array([
            entrance['X'],
            entrance['Y'],
            entrance['BB Width'],
            entrance['BB Height'],
            bb_type + 1
        ]).T
        img_data[filename]['bbs'].append(bb)
    return img_data

def slice_dict(dictionary, l, r):
    keys = list(dictionary.keys())
    keys = keys[l:r]
    ret_dictionary = {}
    for key in keys:
        ret_dictionary[key] = dictionary[key]
    return ret_dictionary

### path to images folder
root_dir = '/Users/fesiib/Desktop/KIXLab/dev/lecture-design-dataset/dataset/images/'
### path to csv file
csv_file = '/Users/fesiib/Desktop/KIXLab/dev/lecture-design-dataset/dataset/ver123_schema.csv'

dataset = pd.read_csv(csv_file)
img_data = process_fitvid_dataset(dataset, root_dir)

division = int(args.train_portion * len(img_data))

train_dataset = FitVidDataset(
    img_data=slice_dict(img_data, 0, division),
    root_dir=root_dir,
    transform = transforms.Compose([
        Rescale(256),
        RandomCrop(128),
        ToTensor(),
    ])
)
test_dataset = FitVidDataset(
    img_data=slice_dict(img_data, division, len(img_data)),
    root_dir=root_dir,
    transform = transforms.Compose([
        Rescale(256),
        RandomCrop(128),
        ToTensor(),
    ])
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)