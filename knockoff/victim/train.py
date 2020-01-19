#!/usr/bin/python
"""Trains and stores the victim model. We use it later to generate transfer set.
"""
import argparse
import json
import os
import os.path as osp
from datetime import datetime

import torch
import torch.nn as nn
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
	parser.add_argument('--datadir', default='.data',
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


	# 20200117 LIN,Y.D. More arguments
	parser.add_argument('--hidden_size', type=int, default=32, metavar='N',
						help='The hidden size for the recurrent network')

	args = parser.parse_args()
	params = vars(args)

	# torch.manual_seed(cfg.DEFAULT_SEED)
	# extract parameter arguments into variables
	embed_dim = params['embed_dim']
	dataset_name = params['dataset']
	datadir = params['datadir']
	out_path = params['out_path']
	batch_size = params['batch_size']
	lr = params['lr']
	lr_gamma = params['lr_gamma']
	num_workers = params['num_workers']
	num_epochs = params['epochs']
	model_name = params['model_arch']
	pretrained = params['pretrained']

	if params['device_id'] >= 0:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
		device = torch.device('cuda:'+ params['device_id'])
	else:
		device = torch.device('cpu')

	# Currently supports only the torchtext datasets
	valid_datasets = list(text_classification.DATASETS.keys())
	if dataset_name not in valid_datasets:
		raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))

	# Currently, we will support classification tasks only. Sentiment task is the next one.
	# See /knockoff/datasets/__init__.py for mapping
	modelfamily = datasets.dataset_to_modelfamily[dataset_name]  # e.g. 'classification'
	metadata = datasets.dataset_metadata[dataset_name]  # Relevant parameters for the task. e.g. 'ngram'

	ngrams = metadata['ngram']

	# If dataset does not exist, download it and save it
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

	# Extract variables for model from the dataset
	vocab_size = len(trainset.get_vocab())
	params['num_classes'] = len(trainset.get_labels())
	num_classes = params['num_classes']
	model = zoo.get_net(model_name, modelfamily, pretrained, 
						vocab_size=vocab_size, embed_dim=embed_dim,
						num_class=num_classes)
	model = model.to(device)

	# 20200117 LIN,Y.D. Conditions for different models
	if model_name in ['attention_model']:
		pass
	elif model_name == 'wordembedding':
		optimizer = torch.optim.SGD(model.parameters(), lr=lr)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=lr_gamma)
		criterion = nn.CrossEntropyLoss(reduction='mean')
		collate_fn = model_utils.generate_batch

	model_utils.train_and_valid(
		trainset, testset, model, model_name, modelfamily, 
		out_path, batch_size, optimizer, scheduler, criterion, 
		lr, lr_gamma, num_workers, collate_fn, num_epochs, device)

	# Store arguments in json file. Maybe for the transfer set step?
	params['created_on'] = str(datetime.now())
	params_out_path = osp.join(out_path, 'params.json')
	with open(params_out_path, 'w') as jf:
		json.dump(params, jf, indent=True)


if __name__ == '__main__':
	main()
