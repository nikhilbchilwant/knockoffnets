#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import json
import os
import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import optim
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

import knockoff.config as cfg
import knockoff.utils.model as model_utils
from knockoff import datasets
import knockoff.models.zoo as zoo
from torchtext.datasets import text_classification
from knockoff.adversary.transfer import TransferSet

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--budgets', metavar='B', type=int)
    # Optional arguments
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    # Attacker's defense
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=True)
    parser.add_argument('--optimizer_choice', type=str, help='Optimizer', default='sgdm', choices=('sgd', 'sgdm', 'adam', 'adagrad'))
    parser.add_argument('--datadir', default='.data', help='data directory (default=.data)')
    parser.add_argument('--embed-dim', type=int, default=32,
                        help='embed dim. (default=32)')
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.MODEL_DIR)

    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model_dir = params['model_dir']

    # ----------- Set up transferset
    transferset_path = osp.join(model_dir, 'transferset.pickle')
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples.num_classes
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples.data), num_classes))

    # ----------- Set up testset
    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]  # e.g. 'classification'
    metadata = datasets.dataset_metadata[dataset_name]  # Relevant parameters for the task. e.g. 'ngram'

    # Currently supports only the torchtext datasets
    valid_datasets = list(text_classification.DATASETS.keys())
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))

    # If dataset does not exist, download it and save it
    datadir = params['datadir']
    ngrams = metadata['ngram']

    dataset_dir = datadir + '/' + dataset_name.lower() + '_csv'
    train_data_path = os.path.join(dataset_dir, dataset_name + "_ngrams_{}_train.data".format(ngrams))
    test_data_path = os.path.join(dataset_dir, dataset_name + "_ngrams_{}_test.data".format(ngrams))
    if not (os.path.exists(train_data_path) and os.path.exists(test_data_path)):
        if not os.path.exists('.data'):
            print("Creating directory {}".format(datadir))
            os.mkdir('.data')
        trainset, testset = text_classification.DATASETS[dataset_name](root='.data', ngrams=ngrams)
        print("Saving train data to {}".format(train_data_path))
        torch.save(trainset, train_data_path)
        print("Saving test data to {}".format(test_data_path))
        torch.save(testset, test_data_path)
    else:
        print("Loading train data from {}".format(train_data_path))
        trainset = torch.load(train_data_path)
        print("Loading test data from {}".format(test_data_path))
        testset = torch.load(test_data_path)

    if len(trainset.get_labels()) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.get_labels())))

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']

    model = zoo.get_net(model_name, modelfamily, pretrained, vocab_size=len(trainset.get_vocab()), embed_dim=params['embed_dim'],
                        num_class=num_classes)

    # ----------- Train
    # budget = params['budgets']
    # assert budget <= len(transferset_samples.data * 8), 'Required {} samples > Found {} samples'.format(budget, len(transferset_samples.data))

    model_utils.train_and_valid_knockoff(transferset_samples.data, testset, model, model_name, modelfamily, device=device, **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
