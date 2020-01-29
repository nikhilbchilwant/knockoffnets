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
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import text_classification
from tqdm import tqdm

import knockoff.config as cfg
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
from knockoff import datasets
from knockoff.victim.blackbox import Blackbox

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


# Stores transfer set as a list of btaches
class TransferSet(object):
	def __init__(self, num_classes, data_batches, vocab_size, batch_size):
		self.num_classes = num_classes
		self.data = data_batches
		self.batch_size = batch_size
		self.vocab_size = vocab_size

class RandomAdversary(object):
	def __init__(self, blackbox, queryset, num_classes, vocab_size, batch_size=16):
		self.blackbox = blackbox
		self.queryset = queryset

		self.n_queryset = len(self.queryset)
		self.batch_size = batch_size
		self.idx_set = set()
		self.transferset = TransferSet(num_classes, None, vocab_size, batch_size)
		# self.transferset = TransferSet(len(self.queryset.get_labels()), None, batch_size)  # List of tuples [(text tensor, label)]

		self._restart()

	def _restart(self):
		np.random.seed(cfg.DEFAULT_SEED)
		torch.manual_seed(cfg.DEFAULT_SEED)
		torch.cuda.manual_seed(cfg.DEFAULT_SEED)

		self.idx_set = set(range(len(self.queryset)))
		# self.transferset = TransferSet(len(self.queryset.get_labels()), None, self.batch_size)

	def get_transferset(self, budget, collate_fn):
		sampler = torch.utils.data.RandomSampler(self.queryset, replacement=True, num_samples=budget)
		count = 0
		# selected_element_indices = np.random.choice(len(self.queryset), budget)

		with tqdm(total=budget) as pbar:
			# while budget>0:
			#     current_batch_size = self.batch_size
			#     if budget < self.batch_size:
			#         current_batch_size = budget
			batch_data = []
			data = DataLoader(self.queryset, batch_size=self.batch_size, 
							  collate_fn=collate_fn, sampler=sampler)
			# data = DataLoader(self.queryset, batch_size=self.batch_size, collate_fn=model_utils.generate_batch, sampler=sampler)
			for i, (text, offsets, label) in enumerate(data):
				query_prediction_probabilities = self.blackbox(text, offsets)
				# offsets = torch.cat((offsets, torch.tensor([len(text)-1])), dim=0)
				labels = query_prediction_probabilities.argmax(1)

				# for sample_index in range(0, len(query_prediction_probabilities)):
				#     self.transferset.append((text.narrow(0, offsets[sample_index],
				#                                          offsets[sample_index + 1] - offsets[sample_index] + 1), offsets,
				#                              query_prediction_probabilities[sample_index]))
				batch_data.append((text, offsets, labels))
				count = count + len(label)
				pbar.update(count)

		self.transferset.data = batch_data
		return self.transferset


def remap_indices(vocab_victim, vocab_adversary):
	
	adv_idx_to_victim_idx = {}

	for i, s in enumerate(vocab_adversary.itos):
		adv_idx_to_victim_idx[i] = vocab_victim[s]

	return adv_idx_to_victim_idx


def main():
	parser = argparse.ArgumentParser(description='Construct transfer set')
	parser.add_argument('policy', metavar='PI', type=str, help='Policy to use while training',
						choices=['random', 'adaptive'])
	parser.add_argument('victim_model_dir', metavar='PATH', type=str,
						help='Path to victim model. Should contain files "model_best.pth.tar" and "params.json"')  # ??? Why do we need this?
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
	parser.add_argument('-w', '--nworkers', metavar='N', type=int, 
						help='# Worker threads to load data', default=10)
	parser.add_argument('-n', '--ngrams', metavar='NG', type=int, help='#n-grams', default=1)
	parser.add_argument('--open_world', metavar='B', type=int, default=1, 
						help='Activate open_world will remap the indices')
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

	folder_name = datasets.dataset_metadata[queryset_name]['alias']
	dataset_dir = dataset_dir + '/' + folder_name + '_csv'
	# dataset_dir = dataset_dir + '/' + queryset_name.lower() + '_csv'

	train_data_path = os.path.join(dataset_dir, queryset_name + "_ngrams_{}_train.data".format(ngrams))
	test_data_path = os.path.join(dataset_dir, queryset_name  + "_ngrams_{}_test.data".format(ngrams))
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

	# ----------- Initialize blackbox i.e. victim model
	blackbox_dir = params['victim_model_dir']
	blackbox, model_arch, seq_len, num_classes = Blackbox.from_modeldir(blackbox_dir, device)
	print('check blackbox num_classes:', num_classes)
	# 20200128 LIN,Y.D. Remap adversary's indices to victim's indices
	queryset, _ = trainset, testset

	if params['open_world']:

		victim_vocab_file = open(osp.join(params['victim_model_dir'], 'stoi.pkl'), 'rb')
		victim_vocab = pickle.load(victim_vocab_file)
		victim_vocab_file.close()

		adversary_vocab = trainset.get_vocab()
		adv_idx_to_victim_idx = remap_indices(victim_vocab, adversary_vocab)

	# ----------- Initialize adversary
	batch_size = params['batch_size']
	nworkers = params['nworkers']
	transfer_out_path = osp.join(out_path, 'transferset.pickle')
	if params['policy'] == 'random':
		adversary = RandomAdversary(blackbox, queryset, num_classes, len(adversary_vocab), 
									batch_size=batch_size)
	elif params['policy'] == 'adaptive':
		raise NotImplementedError()
	else:
		raise ValueError("Unrecognized policy")

	print('=> constructing transfer set...')
	if model_arch in ['attention_model', 'self_attention', 'rcnn']:

		if params['open_world']:
			collate_fn = partial(model_utils.generate_batch_for_var_length_with_remapped_indices,
								 adv_idx_to_victim_idx,
								 seq_len)
		else:
			collate_fn = partial(model_utils.generate_batch_for_var_length, seq_len)

	elif model_arch == 'wordembedding':

		if params['open_world']:
			collate_fn = partial(model_utils.generate_batch_with_remapped_indices,
								 adv_idx_to_victim_idx,
								 seq_len)
		else:
			collate_fn = partial(model_utils.generate_batch, seq_len)

		# transferset = adversary.get_transferset(params['budget'], collate_fn)
	else:
		raise ValueError('No architecture support.')

	transferset = adversary.get_transferset(params['budget'], collate_fn)

	print('check transferset.num_classes:', transferset.num_classes)

	with open(transfer_out_path, 'wb') as wf:
		pickle.dump(transferset, wf)
	print('=> transfer set ({} samples) written to: {}'.format(
		len(transferset.data) * transferset.batch_size, transfer_out_path))

	# Store arguments
	params['created_on'] = str(datetime.now())
	params_out_path = osp.join(out_path, 'params_transfer.json')
	with open(params_out_path, 'w') as jf:
		json.dump(params, jf, indent=True)


if __name__ == '__main__':
	main()