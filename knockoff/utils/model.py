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


def train_and_valid(trainset, testset, model, model_name, modelfamily, out_path, batch_size, lr, lr_gamma, num_workers,
					num_epochs=5, device='cpu'):
	if not osp.exists(out_path):
		knockoff_utils.create_dir(out_path)

	optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=lr_gamma)
	criterion = nn.CrossEntropyLoss(reduction='mean')
	train_data = DataLoader(trainset, batch_size=batch_size, shuffle=True,
							collate_fn=generate_batch, num_workers=num_workers)
	num_lines = num_epochs * len(train_data)

	best_test_acc, test_acc = -1., -1.

	# Initialize logging
	log_path = osp.join(out_path, 'train-{}-{}.log.tsv'.format(model_name, modelfamily))
	if not osp.exists(log_path):
		with open(log_path, 'w') as wf:
			columns = ['run_id', 'epoch', 'training loss', 'test accuracy', 'best_accuracy']
			wf.write('\t'.join(columns) + '\n')

	model_out_path = osp.join(out_path, 'checkpoint-{}-{}.pth.tar'.format(model_name, modelfamily))

	for epoch in range(num_epochs):

		for i, (text, offsets, cls) in enumerate(train_data):
			optimizer.zero_grad()
			text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
			output = model(text, offsets)
			train_loss = criterion(output, cls)
			train_loss.backward()
			optimizer.step()
			processed_lines = i + len(train_data) * epoch
			progress = processed_lines / float(num_lines)
			if processed_lines % 128 == 0:
				sys.stderr.write(
					"\rTraining progress: {:3.0f}% lr: {:3.3f} loss: {:3.3f}".format(
						progress * 100, scheduler.get_lr()[0], train_loss))

		scheduler.step()

		print("")
		test_acc = test(model, testset)
		best_test_acc = max(best_test_acc, test_acc)
		print("Test - Accuracy: {}".format(test_acc))

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


def test(model, test_data, batch_size=16, device='cpu'):
	data = DataLoader(test_data, batch_size=batch_size, collate_fn=generate_batch)
	total_accuracy = []
	for text, offsets, cls in data:
		text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
		with torch.no_grad():
			output = model(text, offsets)
			accuracy = (output.argmax(1) == cls).float().mean().item()
			total_accuracy.append(accuracy)

	if total_accuracy == []:
		return 0.0

	return sum(total_accuracy) / len(total_accuracy)

def train_and_valid_knockoff(trainset, testset, model, model_name, model_family, batch_size=64, 	criterion_train=None, criterion_test=None, device=None, num_workers=10, momentum=0.5,
	lr_step=30, resume=None, log_interval=100, weighted_loss=False,	checkpoint_suffix='', optimizer=None, scheduler=None, **kwargs):
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
		test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	else:
		test_loader = None

	if weighted_loss:
		if not isinstance(trainset.samples[0][1], int):
			print('Labels in trainset is of type: {}. Expected: {}.'.format(type(trainset.samples[0][1]), int))

		class_to_count = dd(int)
		for _, y in trainset.samples:
			class_to_count[y] += 1
		class_sample_count = [class_to_count[c] for c, cname in enumerate(trainset.classes)]
		print('=> counts per class: ', class_sample_count)
		weight = np.min(class_sample_count) / torch.Tensor(class_sample_count)
		weight = weight.to(device)
		print('=> using weights: ', weight)
	else:
		weight = None

	# Optimizer
	if criterion_train is None:
		criterion = nn.CrossEntropyLoss(reduction='mean')
	if criterion_test is None:
		criterion = nn.CrossEntropyLoss(reduction='mean')
	if optimizer is None:
		optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	if scheduler is None:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=lr_gamma)

	model_out_path = osp.join(out_path, 'checkpoint-{}-{}.pth.tar'.format(model_name, model_family))

	num_lines = num_epochs * len(trainset)
	best_test_acc, test_acc = -1., -1.

	for epoch in range(num_epochs):

		for i, (text, offsets, cls) in enumerate(trainset):
			optimizer.zero_grad()
			text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
			output = model(text, offsets)
			train_loss = criterion(output, cls)
			train_loss.backward()
			optimizer.step()
			processed_lines = i + len(trainset) * epoch
			progress = processed_lines / float(num_lines)
			if processed_lines % 128 == 0:
				sys.stderr.write(
					"\rTraining progress: {:3.0f}% lr: {:3.3f} loss: {:3.3f}".format(
						progress * 100, scheduler.get_lr()[0], train_loss))

		scheduler.step()

		print("")
		test_acc = test(model, testset)
		best_test_acc = max(best_test_acc, test_acc)
		print("Test - Accuracy: {}".format(test_acc))

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
