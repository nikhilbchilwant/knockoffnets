#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
import pickle
import json
from datetime import datetime

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision

from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
from knockoff.victim.blackbox import Blackbox
import knockoff.config as cfg

from torchtext.datasets import text_classification

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class RandomAdversary(object):
    def __init__(self, blackbox, queryset, batch_size=16):
        self.blackbox = blackbox
        self.queryset = queryset

        self.n_queryset = len(self.queryset)
        self.batch_size = batch_size
        self.idx_set = set()

        self.transferset = []  # List of tuples [(img_path, output_probs)]

        self._restart()

    def _restart(self):
        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        self.idx_set = set(range(len(self.queryset)))
        self.transferset = []

    def get_transferset(self, budget):

        sampler = torch.utils.data.RandomSampler(self.queryset, replacement=True, num_samples=budget)
        count = 0
        # selected_element_indices = np.random.choice(len(self.queryset), budget)
        with tqdm(total=budget) as pbar:
            # while budget>0:
            #     current_batch_size = self.batch_size
            #     if budget < self.batch_size:
            #         current_batch_size = budget

            data = DataLoader(self.queryset, batch_size=self.batch_size, collate_fn=model_utils.generate_batch, sampler=sampler)
            for text, offsets, cls in data:
                query_prediction_probabilities = self.blackbox(text, offsets)
                predicted_label = query_prediction_probabilities.argmax(1)
                for text_sample, predicted_label_sample in zip(text, predicted_label.cpu()):
                    self.transferset.append((text_sample, predicted_label_sample.cpu()))
                count = count + len(cls)
                pbar.update(count)

            # budget = budget - self.batch_size
            #     count = count + current_batch_size
            # pbar.update(count)
                # with tqdm(total=budget) as pbar:
                #     for t, B in enumerate(range(start_B, end_B, self.batch_size)):
                #         idxs = np.random.choice(list(self.idx_set), replace=False,
                #                                 size=min(self.batch_size, budget - len(self.transferset)))
                #         self.idx_set = self.idx_set - set(idxs)
                #
                #         if len(self.idx_set) == 0:
                #             print('=> Query set exhausted. Now repeating input examples.')
                #             self.idx_set = set(range(len(self.queryset)))
                #
                #         x_t = torch.stack([self.queryset[i][0] for i in idxs]).to(self.blackbox.device)
                #         y_t = self.blackbox(x_t).cpu()
                #
                #         if hasattr(self.queryset, 'samples'):
                #             # Any DatasetFolder (or subclass) has this attribute
                #             # Saving image paths are space-efficient
                #             img_t = [self.queryset.samples[i][0] for i in idxs]  # Image paths
                #         else:
                #             # Otherwise, store the image itself
                #             # But, we need to store the non-transformed version
                #             img_t = [self.queryset.data[i] for i in idxs]
                #             if isinstance(self.queryset.data[0], torch.Tensor):
                #                 img_t = [x.numpy() for x in img_t]
                #
                #         for i in range(x_t.size(0)):
                #             img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                #             self.transferset.append((img_t_i, y_t[i].cpu().squeeze()))
                #
                #         pbar.update(x_t.size(0))

        return self.transferset


def main():
    parser = argparse.ArgumentParser(description='Construct transfer set')
    parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training',
                        choices=['random', 'adaptive'])
    parser.add_argument('victim_model_dir', metavar='PATH', type=str,
                        help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"') #??? Why do we need this?
    parser.add_argument('--out_dir', metavar='PATH', type=str,
                        help='Destination directory to store transfer set', required=True)
    parser.add_argument('--budget', metavar='N', type=int, help='Size of transfer set to construct',
                        required=True)
    parser.add_argument('--queryset', metavar='TYPE', type=str, help='Adversary\'s dataset (P_A(X))', required=True)
    parser.add_argument('--batch_size', metavar='TYPE', type=int, help='Batch size of queries', default=8)
    # parser.add_argument('--topk', metavar='N', type=int, help='Use posteriors only from topk classes',
    #                     default=None)
    # parser.add_argument('--rounding', metavar='N', type=int, help='Round posteriors to these many decimals',
    #                     default=None)
    # parser.add_argument('--tau_data', metavar='N', type=float, help='Frac. of data to sample from Adv data',
    #                     default=1.0)
    # parser.add_argument('--tau_classes', metavar='N', type=float, help='Frac. of classes to sample from Adv data',
    #                     default=1.0)
    # ----------- Other params
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id', default=0)
    parser.add_argument('-w', '--nworkers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('-n', '--ngrams', metavar='NG', type=int, help='#n-grams', default=2)
    args = parser.parse_args()
    params = vars(args)

    out_path = params['out_dir']
    knockoff_utils.create_dir(out_path)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up queryset
    queryset_name = params['queryset']
    ngrams = params['ngrams']
    valid_datasets = list(text_classification.DATASETS.keys())
    if queryset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name]
    # transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    # queryset = datasets.__dict__[queryset_name](train=True, transform=transform)
    dataset_dir = '.data'
    dataset_dir = dataset_dir + '/' + queryset_name.lower() + '_csv'
    train_data_path = os.path.join(dataset_dir, queryset_name + "_ngrams_{}_train.data".format(ngrams))
    test_data_path = os.path.join(dataset_dir, queryset_name + "_ngrams_{}_test.data".format(ngrams))
    if not (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
        if not os.path.exists('.data'):
            print("Creating directory {}".format(dataset_dir))
            os.mkdir('.data')
        trainset, testset = text_classification.DATASETS[queryset_name](root='.data', ngrams=ngrams)
        print("Saving train data to {}".format(train_data_path))
        torch.save(trainset, train_data_path)
        print("Saving test data to {}".format(test_data_path))
        torch.save(testset, test_data_path)
    else:
        print("Loading train data from {}".format(train_data_path))
        trainset = torch.load(train_data_path)
        print("Loading test data from {}".format(test_data_path))
        testset = torch.load(test_data_path)

    queryset, _ = trainset, testset
    vocab_size = len(trainset.get_vocab())
    num_classes = len(trainset.get_labels())
    # ----------- Initialize blackbox i.e. victim model
    blackbox_dir = params['victim_model_dir']
    embed_dim = 32
    blackbox = Blackbox.from_modeldir(blackbox_dir, vocab_size, num_classes, embed_dim, device)

    # ----------- Initialize adversary
    batch_size = params['batch_size']
    nworkers = params['nworkers']
    transfer_out_path = osp.join(out_path, 'transferset.pickle')
    if params['policy'] == 'random':
        adversary = RandomAdversary(blackbox, queryset, batch_size=batch_size)
    elif params['policy'] == 'adaptive':
        raise NotImplementedError()
    else:
        raise ValueError("Unrecognized policy")

    print('=> constructing transfer set...')
    transferset = adversary.get_transferset(params['budget'])
    with open(transfer_out_path, 'wb') as wf:
        pickle.dump(transferset, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transfer_out_path))

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(out_path, 'params_transfer.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
