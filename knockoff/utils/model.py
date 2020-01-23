#!/usr/bin/python
"""
Common methods used by model
"""

import os.path as osp
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import knockoff.utils.utils as knockoff_utils
import torch.nn.functional as F
from collections import defaultdict as dd
import numpy as np
import time

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def generate_batch(batch):
	label = torch.tensor([entry[0] for entry in batch])
	text = [entry[1] for entry in batch]
	offsets = [0] + [len(entry) for entry in text]

	offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
	text = torch.cat(text)

	return text, offsets, label


def generate_batch_for_var_length(batch):

	label = torch.tensor([entry[0] for entry in batch])
	text = [entry[1] for entry in batch]
	padded_text = pad_sequence(text, padding_value=1)
	text_lengths = torch.tensor(
		sorted([len(t) for t in text], reverse=True))

	return padded_text, text_lengths, label


def train_and_valid(trainset, testset, model, model_name, modelfamily, 
					out_path, batch_size, optimizer, scheduler, criterion, 
					lr, lr_gamma, num_workers, collate_fn, 
					num_epochs=5, device='cpu'):
	if not osp.exists(out_path):
		knockoff_utils.create_dir(out_path)
		print('Created following directory: ', osp.abspath(out_path))

	train_data = DataLoader(trainset, batch_size=batch_size, shuffle=True,
							collate_fn=collate_fn, num_workers=num_workers)
	num_lines = num_epochs * len(train_data)
	pbar = tqdm(total=num_lines)

	best_test_acc, test_acc = -1., -1.

	# Initialize logging
	log_path = osp.join(out_path, 'train-{}-{}.log.tsv'.format(model_name, modelfamily))
	if not osp.exists(log_path):
		with open(log_path, 'w') as wf:
			columns = ['run_id', 'epoch', 'training loss', 'test accuracy', 'best_accuracy']
			wf.write('\t'.join(columns) + '\n')

	model_out_path = osp.join(out_path, 'checkpoint.pth.tar'.format(model_name, modelfamily))

	for epoch in range(num_epochs):

		total = 0
		train_acc = 0

		model.train()

		for i, (text, textmeta, lbl) in enumerate(train_data):
			optimizer.zero_grad()
			text, textmeta, lbl = \
				text.to(device), textmeta.to(device), lbl.to(device)
			output = model(text, textmeta)
			train_loss = criterion(output, lbl)
			train_loss.backward()
			optimizer.step()
			pbar.update(1)

			# processed_lines = i + len(train_data) * epoch
			# progress = processed_lines / float(num_lines)
			# if processed_lines % 128 == 0:
			# 	sys.stderr.write(
			# 		"\rTraining progress: {:3.0f}% lr: {:3.5f} loss: {:3.3f}".format(
			# 			progress * 100, scheduler.get_lr()[0], train_loss))

			_, pred = torch.max(output.data, 1)
			train_acc += (lbl == pred).sum().item()
			total += len(lbl)

		scheduler.step()
		train_acc /= total

		test_acc, test_loss = test(model, criterion, testset, batch_size, collate_fn, device)
		best_test_acc = max(best_test_acc, test_acc)
		print("")
		print("Train acc: {:3.3f}, Train loss: {:3.3f}, Valid acc: {:3.3f}, Valid loss: {:3.3f}".format(
			train_acc, train_loss, test_acc, test_loss))
		

		if test_acc >= best_test_acc:
			state = {
				'epoch': epoch,
				'arch': model.__class__,
				'state_dict': model.state_dict(),
				'best_acc': test_acc,
				'optimizer': optimizer.state_dict(),
				'created_on': str(datetime.now()),
			}
			torch.save(state, model_out_path)

		# Log
		run_id = str(datetime.now())
		with open(log_path, 'a') as af:
			data_column = [run_id, epoch, train_loss.item(), test_acc, best_test_acc]
			af.write('\t'.join([str(c) for c in data_column]) + '\n')

	# Select the best model for prediction
	print('Select the model from epoch {}'.format(state['epoch']))
	model.load_state_dict(state['state_dict'])


def test(model, criterion, test_data, batch_size, collate_fn, device='cpu'):
	data = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
	total_accuracy = []
	test_loss = .0
	model.eval()
	for text, textmeta, lbl in data:
		text, textmeta, lbl = text.to(device), textmeta.to(device), lbl.to(device)
		with torch.no_grad():
			output = model(text, textmeta)
			loss = criterion(output, lbl)
			accuracy = (output.argmax(1) == lbl).float().mean().item()
			total_accuracy.append(accuracy)
			test_loss += loss.item()

	if total_accuracy == []:
		return 0.0

	test_acc = sum(total_accuracy) / len(total_accuracy)
	test_loss /= len(data)
	return test_acc, test_loss


def train_and_valid_knockoff(trainset, testset, model, model_name, model_family, 
							 batch_size=64, device=None, num_workers=10, collate_fn=generate_batch,
							 optimizer=None, criterion=None, scheduler=None, checkpoint_suffix='', 
							 **kwargs):

	out_path = kwargs['model_dir']
	lr = kwargs['lr']
	lr_gamma = kwargs['lr_gamma']
	num_epochs = kwargs['epochs']
	if device is None:
		device = torch.device('cuda')
	if not osp.exists(out_path):
		knockoff_utils.create_dir(out_path)
	run_id = str(datetime.now())

	# Data loaders
	# trainset = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	if testset is not None:
		test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
								 num_workers=num_workers, collate_fn=collate_fn)
	else:
		test_loader = None

	# if weighted_loss:
	# 	if not isinstance(trainset.samples[0][1], int):
	# 		print('Labels in trainset is of type: {}. Expected: {}.'.format(
	# 			type(trainset.samples[0][1]), int))

	# 	class_to_count = dd(int)
	# 	for _, y in trainset.samples:
	# 		class_to_count[y] += 1
	# 	class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
	# 	print('=> counts per class: ', class_sample_count)
	# 	weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
	# 	weight = weight.to(device)
	# 	print('=> using weights: ', weight)
	# else:
	# 	weight = None

	num_lines = num_epochs * len(trainset)
	pbar = tqdm(total=num_lines)

	model_out_path = osp.join(out_path, 'checkpoint-{}-{}.pth.tar'.format(model_name, model_family))

	num_lines = num_epochs * len(trainset)
	best_test_acc, test_acc = -1., -1.

	for epoch in range(num_epochs):

		total = 0
		train_acc = .0
		model.train()

		for i, (text, textmeta, lbl) in enumerate(trainset):
			optimizer.zero_grad()
			text, textmeta, lbl = text.to(device), textmeta.to(device), lbl.to(device)
			output = model(text, textmeta)
			train_loss = criterion(output, lbl)
			train_loss.backward()
			optimizer.step()
			pbar.update(1)
			_, pred = torch.max(output.data, 1)
			train_acc += (lbl == pred).sum().item()
			total += len(lbl)

		scheduler.step()
		train_acc /= total
		train_loss = train_loss.item()
		test_acc, test_loss = test(model, criterion, testset, batch_size, collate_fn, device)
		best_test_acc = max(best_test_acc, test_acc)

		print("Train acc: {:3.3f}, Train loss: {:3.3f}, Valid acc: {:3.3f}, Valid loss: {:3.3f}".format(
			train_acc, train_loss, test_acc, test_loss))
		print("")

		if test_acc >= best_test_acc:
			state = {
				'epoch': epoch,
				'arch': model.__class__,
				'state_dict': model.state_dict(),
				'best_acc': test_acc,
				'optimizer': optimizer.state_dict(),
				'created_on': str(datetime.now()),
			}
			torch.save(state, model_out_path)

	return model

