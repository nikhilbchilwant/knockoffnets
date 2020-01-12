#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import json
import os
import os.path as osp
from datetime import datetime

import torch
from torchtext.datasets import text_classification

import knockoff.config as cfg
import knockoff.models.zoo as zoo
import knockoff.utils.model as model_utils
from knockoff import datasets

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"

"""
Trains a model and emits model parameters from the given dataset and selected model architecture.
"""


def main():
	parser = argparse.ArgumentParser(description='Train a model')
	# Required arguments
	parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
	parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
	# Optional arguments
	parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
						default=cfg.MODEL_DIR)
	parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
	parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
						help='input batch size for training (default: 64)')
	parser.add_argument('--datadir', default='./.data',
						help='data directory (default=.data)')
	parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
						help='number of epochs to train (default: 100)')
	parser.add_argument('--embed-dim', type=int, default=32,
						help='embed dim. (default=32)')
	parser.add_argument('--lr', type=float, default=4.0, metavar='LR',
						help='learning rate (default: 0.1)')
	# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
	# 					help='SGD momentum (default: 0.5)')
	parser.add_argument('--log-interval', type=int, default=100, metavar='N',
						help='how many batches to wait before logging training status')
	parser.add_argument('--resume', default=None, type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')
	parser.add_argument('--lr-step', type=int, default=1, metavar='N',
						help='Step sizes for LR')
	parser.add_argument('--lr-gamma', type=float, default=0.8, metavar='N',
						help='LR Decay Rate')
	parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
	parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
	parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
	parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=None)

	args = parser.parse_args()
	params = vars(args)

	# torch.manual_seed(cfg.DEFAULT_SEED)
	if params['device_id'] >= 0:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	embed_dim = params['embed_dim']
	# ----------- Set up dataset
	# Checks validity of provided parameters
	dataset_name = params['dataset']
	datadir = params['datadir']
	out_path = params['out_path']
	batch_size = params['batch_size']
	lr = params['lr']
	# lr_step = params['lr-step']
	lr_gamma = params['lr_gamma']
	num_workers = params['num_workers']
	num_epochs = params['epochs']

	# valid_datasets = datasets.__dict__.keys() #gives: dict_keys(['__file__', 'SVHN', '__spec__', '__cached__', '__path__', 'cifarlike', 'modelfamily_to_transforms', 'diabetic5', 'KMNIST', 'caltech256', 'Caltech256', '__doc__', 'dataset_to_modelfamily', 'Indoor67', 'transforms', '__package__', 'CIFAR10', 'EMNISTLetters', '__name__', 'indoor67', 'tinyimagenet200', 'Diabetic5', 'EMNIST', 'mnistlike', 'MNIST', '__builtins__', 'TinyImageNet200', 'FashionMNIST', 'ImageNet1k', 'imagenet1k', 'CIFAR100', 'cubs200', 'CUBS200', '__loader__'])
	valid_datasets = list(text_classification.DATASETS.keys())
	if dataset_name not in valid_datasets:
		raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
	# dataset = datasets.__dict__[dataset_name]


	# TEXT = data.Field(lower=True, use_vocab=True, tokenizer_language='en')
	# LABEL = data.Field(sequential=False, use_vocab=False)
	# data_fields = [
	# 	('NewsType', LABEL), ('NewsHeading', TEXT)  # , ('News', TEXT)
	# ]
	# trainset, testset = data.TabularDataset.splits(path=datadir, train='train.csv', test='test.csv', format='csv',
	# 											   fields=data_fields)
	# vec = vocab.Vectors('glove.6B.100d.txt', '../../data/glove_embedding/')
	# TEXT.build_vocab(trainset, vectors=vec)
	# LABEL.build_vocab(trainset)
	modelfamily = datasets.dataset_to_modelfamily[dataset_name]
	metadata = datasets.dataset_metadata[dataset_name]
	# train_transform = datasets.modelfamily_to_transforms[modelfamily]['train'] #??? Is it right to use imagenet transformer for all other datasets in the family?
	# test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
	# trainset = dataset(train=True)
	# testset = dataset(train=False)
	# num_classes = metadata['num_classes']

	# if params['train_subset'] is not None:
	# 	idxs = np.arange(len(trainset))
	# 	ntrainsubset = params['train_subset']
	# 	idxs = np.random.choice(idxs, size=ntrainsubset, replace=False)
	# 	trainset = Subset(trainset, idxs)

	# ----------- Set up model
	model_name = params['model_arch']
	pretrained = params['pretrained']
	ngrams = metadata['ngram']

	#Use predownloaded dataset if available
	datadir = datadir + '/' + dataset_name.lower() + '_csv'
	train_data_path = os.path.join(datadir, dataset_name + "_ngrams_{}_train.data".format(ngrams))
	test_data_path = os.path.join(datadir, dataset_name + "_ngrams_{}_test.data".format(ngrams))
	if not os.path.exists(datadir):
		print("Creating directory {}".format(datadir))
		os.mkdir(datadir)
		trainset, testset = text_classification.DATASETS[dataset_name](root=datadir, ngrams=ngrams)
		print("Saving train data to {}".format(train_data_path))
		torch.save(trainset, train_data_path)
		print("Saving test data to {}".format(test_data_path))
		torch.save(testset, test_data_path)
	else:
		# datadir = datadir + '/' + dataset_name.lower() + '_csv'  # path to already present dataset
		print("Loading train data from {}".format(train_data_path))
		trainset = torch.load(train_data_path)
		print("Loading test data from {}".format(test_data_path))
		testset = torch.load(test_data_path)

	vocab_size = len(trainset.get_vocab())
	params['num_classes'] = len(trainset.get_labels())
	num_classes = params['num_classes']

	# model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
	model = zoo.get_net(model_name, modelfamily, pretrained, vocab_size=vocab_size, embed_dim=embed_dim,
						num_class=num_classes)
	model = model.to(device)

	# ----------- Train
	model_utils.train_and_valid(trainset, testset, model, model_name, modelfamily, out_path, batch_size, lr, lr_gamma,
								num_workers, device=device, num_epochs=num_epochs)
	# Store arguments in json file
	params['created_on'] = str(datetime.now())
	params_out_path = osp.join(out_path, 'params.json')
	with open(params_out_path, 'w') as jf:
		json.dump(params, jf, indent=True)


if __name__ == '__main__':
	main()
