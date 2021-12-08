import os
import numpy as np
import pandas as pd

from functools import cmp_to_key

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_BB_types, get_args, get_bb_types

# Basic settings
torch.manual_seed(470)
torch.cuda.manual_seed(470)

BB_TYPES = get_bb_types()
args = get_args()

class BBSlideDeckDataset(Dataset):
    """ Slide Deck Dataset but with Bounding Boxes"""
    def __init__(self, slide_deck_data, slide_deck_N, transform=None):
        self.transform = transform

        self.slide_deck_data = {}

        for key, val in slide_deck_data.items():
            n = len(val['slides'])
            for i in range(n):
                if (n - i < slide_deck_N):
                    break
                if (i % slide_deck_N == 0 or (n - i) == slide_deck_N):
                    current_slide_deck = {
                        'slides': val['slides'][i:(i+slide_deck_N)],
                        'shape': val['shape'],
                    }
                    self.slide_deck_data[str(key) + '_' + str(i)] = current_slide_deck
        print(len(self.slide_deck_data))
        self.slide_deck_ids = list(self.slide_deck_data.keys())
    
    def __len__(self):
        return len(self.slide_deck_ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        slide_deck_id = self.slide_deck_ids[idx]
        (h, w) = self.slide_deck_data[slide_deck_id]["shape"]
        lengths_slide_deck = []
        
        slides = []
        max_len_bbs = args.max_seq_length
        for slide in self.slide_deck_data[slide_deck_id]["slides"]:
            lengths_slide_deck.append(min(max_len_bbs, len(slide)))
            np_slide = np.zeros((max_len_bbs, 5), dtype=np.double)
            for i, bb in enumerate(slide):
                if (i >= max_len_bbs):
                    break
                np_slide[i] = bb
            slides.append(np_slide)
        ref_slide = slides[0]
        slide_deck = slides[1:]
        length_ref_types = lengths_slide_deck.pop(0)
        sample = {
            "shape": (h, w),
            "ref_slide": ref_slide,
            "ref_types": get_BB_types(ref_slide),
            "slide_deck": np.asarray(slide_deck),
            "lengths_slide_deck": lengths_slide_deck,
            "length_ref_types": length_ref_types,
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

class RescaleBB(object):
    """Rescale the bounding boxes in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def _resize_single_slide(self, slide, original_shape, new_shape):
        h, w = original_shape
        new_h, new_w = new_shape
        slide = slide * np.array([new_w / w, new_h / h, new_w / w, new_h / h, 1]).T
        return slide

    def __call__(self, sample):
        h, w = sample["shape"]
        ref_slide = sample["ref_slide"]
        ref_types = sample["ref_types"]
        slide_deck = sample["slide_deck"]
        lengths_slide_deck = sample["lengths_slide_deck"]
        length_ref_types = sample["length_ref_types"]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        ref_slide = self._resize_single_slide(ref_slide, (h, w), (new_h, new_w))
        for i, slide in enumerate(slide_deck):
            slide_deck[i] = self._resize_single_slide(slide, (h, w), (new_h, new_w))

        return {
            "shape": (new_h, new_w),
            "ref_slide": ref_slide,
            "ref_types": ref_types,
            "slide_deck": slide_deck,
            "lengths_slide_deck": lengths_slide_deck,
            "length_ref_types": length_ref_types,
        }

class LeaveN(object):
    def __init__ (self, N):
        self.N = N

    def __call__(self, sample):
        h, w = sample["shape"]
        ref_slide = sample['ref_slide']
        ref_types = sample["ref_types"]
        slide_deck = sample["slide_deck"]
        lengths_slide_deck = sample["lengths_slide_deck"]
        length_ref_types = sample["length_ref_types"]

        if slide_deck.shape[0] > self.N:
            slide_deck = np.delete(slide_deck, range(self.N, slide_deck.shape[0]), 0)
            lengths_slide_deck = lengths_slide_deck[:self.N]

        return {
            "shape": (h, w),
            "ref_slide": ref_slide,
            "ref_types": ref_types,
            "slide_deck": slide_deck,
            "lengths_slide_deck": lengths_slide_deck,
            "length_ref_types": length_ref_types,
        }

class ShuffleRefSlide(object):
    def __call__(self, sample):
        h, w = sample["shape"]
        ref_slide = sample['ref_slide']
        ref_types = sample["ref_types"]
        slide_deck = sample["slide_deck"]
        lengths_slide_deck = sample["lengths_slide_deck"]
        length_ref_types = sample["length_ref_types"]

        lengths_slide_deck.append(length_ref_types)
        slide_deck = np.vstack((slide_deck, ref_slide[None, :]))

        idxs = np.array([*range(0, len(lengths_slide_deck))], dtype=np.int32)
        np.random.shuffle(idxs)

        slide_deck = slide_deck[idxs]

        lengths_slide_deck = np.array(lengths_slide_deck, dtype=np.int32)
        lengths_slide_deck = lengths_slide_deck[idxs]
        lengths_slide_deck = lengths_slide_deck.tolist()
        
        slide_deck = slide_deck.tolist()
        ref_slide = np.asarray(slide_deck.pop())
        length_ref_types = lengths_slide_deck.pop()
        ref_types = get_BB_types(ref_slide)

        slide_deck = np.asarray(slide_deck)
        
        return {
            "shape": (h, w),
            "ref_slide": ref_slide,
            "ref_types": ref_types,
            "slide_deck": slide_deck,
            "lengths_slide_deck": lengths_slide_deck,
            "length_ref_types": length_ref_types,
        }

class ToTensorBB(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        h, w = sample["shape"]
        ref_slide = sample["ref_slide"]
        ref_types = sample["ref_types"]
        slide_deck = sample["slide_deck"]
        lengths_slide_deck = sample["lengths_slide_deck"]
        length_ref_types = sample["length_ref_types"]

        idxs = [*range(0, len(lengths_slide_deck))]

        def by_length(p1, p2):
            return lengths_slide_deck[p2] - lengths_slide_deck[p1]
        idxs = sorted(idxs, key=cmp_to_key(by_length))

        shape = torch.tensor([h, w], dtype=torch.float64)
        ref_slide = torch.from_numpy(ref_slide).float()
        ref_types = torch.from_numpy(ref_types).float()
        
        slide_deck = torch.from_numpy(slide_deck).float()
        lengths_slide_deck = torch.tensor(lengths_slide_deck, dtype=torch.int32)
        
        slide_deck = slide_deck[idxs]
        lengths_slide_deck = lengths_slide_deck[idxs]

        length_ref_types = torch.tensor(length_ref_types, dtype=torch.int32)

        return {
            "shape": shape,
            "ref_slide": ref_slide,
            "ref_types": ref_types,
            "slide_deck": slide_deck,
            "lengths_slide_deck": lengths_slide_deck,
            "length_ref_types": length_ref_types
        }

def process_slide_deck_dataset(all_dataset):
    slide_deck_data = {}
    for entrance in all_dataset.iloc:
        slide_deck_id = entrance['Slide Deck Id']
        
        slide_id = entrance["Slide Id"]
        if (slide_deck_id not in slide_deck_data):
            slide_deck_data[slide_deck_id] = {
                'slides': {},
                'shape': (entrance['Image Height'], entrance['Image Width'])
            }
        
        if slide_id not in slide_deck_data[slide_deck_id]["slides"]:
            slide_deck_data[slide_deck_id]["slides"][slide_id] = []
        bb_type = BB_TYPES.index(entrance['Type'])
        if (bb_type < 0 or bb_type >= len(BB_TYPES)):
            bb_type = len(BB_TYPES)

        bb = np.array([
            entrance['X'],
            entrance['Y'],
            entrance['BB Width'],
            entrance['BB Height'],
            bb_type
        ]).T
        slide_deck_data[slide_deck_id]['slides'][slide_id].append(bb)
    for key in slide_deck_data.keys():
        
        # if key == 100:
        #     for (id, value) in slide_deck_data[key]["slides"].items():
        #         print(56, id)
        #         draw_bbs(slide_deck_data[key]["shape"], value)

        values = list(slide_deck_data[key]["slides"].values())
        slide_deck_data[key]["slides"] = [np.asarray(value) for value in values]
    return slide_deck_data

def slice_dict(dictionary, l, r):
    keys = list(dictionary.keys())
    keys = keys[l:r]
    ret_dictionary = {}
    for key in keys:
        ret_dictionary[key] = dictionary[key]
    return ret_dictionary

def init_dataset():
    print(os.path.dirname(os.getcwd()))
    csv_files_root = os.path.join('./', "data", "bbs")

    dataset = None

    for _, _, files in os.walk(csv_files_root):
        if dataset is not None:
            break
        for file in files:
            if dataset is not None:
                break
            if file.endswith('.csv'):
                print('file: ', file)
                csv_file_path = os.path.join(csv_files_root, file)
                cur_dataset = pd.read_csv(csv_file_path)
                if dataset is None:
                    dataset = cur_dataset
                else:
                    dataset = pd.concat([dataset, cur_dataset])
    slide_deck_data = process_slide_deck_dataset(dataset)

    division = int(args.train_portion * len(slide_deck_data))

    train_slide_deck_dataset = BBSlideDeckDataset(
        slide_deck_data=slice_dict(slide_deck_data, 0, division),
        slide_deck_N=args.slide_deck_N+1,
        transform=transforms.Compose([
            RescaleBB((1, 1)),
            ShuffleRefSlide(),
            LeaveN(args.slide_deck_N),
            ToTensorBB()
        ])
    )

    test_slide_deck_dataset = BBSlideDeckDataset(
        slide_deck_data=slice_dict(slide_deck_data, division, len(slide_deck_data)),
        slide_deck_N=args.slide_deck_N+1,
        transform=transforms.Compose([
            RescaleBB((1, 1)),
            ShuffleRefSlide(),
            LeaveN(args.slide_deck_N),
            ToTensorBB()
        ])
    )

    return (
        train_slide_deck_dataset,
        test_slide_deck_dataset,
    )
