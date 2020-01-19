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


def train_and_valid(trainset, testset, model, model_name, modelfamily, out_path, batch_size=8, lr=4.0, lr_gamma=0.9, num_workers=10,
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


def train_model(model, trainset, batch_size=64, criterion_train=None, criterion_test=None, testset=None,
				device=None, num_workers=10, lr=0.1, momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None,
				epochs=100, log_interval=100, weighted_loss=False, checkpoint_suffix='', optimizer=None, scheduler=None,
				**kwargs):
	out_path = kwargs['model_dir']

	if device is None:
		device = torch.device('cuda')
	if not osp.exists(out_path):
		knockoff_utils.create_dir(out_path)
	run_id = str(datetime.now())

	# Data loaders
	train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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
		criterion_train = nn.CrossEntropyLoss(reduction='mean', weight=weight)
	if criterion_test is None:
		criterion_test = nn.CrossEntropyLoss(reduction='mean', weight=weight)
	if optimizer is None:
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
	if scheduler is None:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
	start_epoch = 1
	best_train_acc, train_acc = -1., -1.
	best_test_acc, test_acc, test_loss = -1., -1., -1.

	# Resume if required
	if resume is not None:
		model_path = resume
		if osp.isfile(model_path):
			print("=> loading checkpoint '{}'".format(model_path))
			checkpoint = torch.load(model_path)
			start_epoch = checkpoint['epoch']
			best_test_acc = checkpoint['best_acc']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(model_path))

	# Initialize logging
	log_path = osp.join(out_path, 'train{}.log.tsv'.format(checkpoint_suffix))
	if not osp.exists(log_path):
		with open(log_path, 'w') as wf:
			columns = ['run_id', 'epoch', 'split', 'loss', 'accuracy', 'best_accuracy']
			wf.write('\t'.join(columns) + '\n')

	model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
	for epoch in range(start_epoch, epochs + 1):
		train_loss, train_acc = train_step(model, train_loader, criterion_train, optimizer, epoch, device,
										   log_interval=log_interval)
		best_train_acc = max(best_train_acc, train_acc)
		scheduler.step(epoch)
		if test_loader is not None:
			test_loss, test_acc = test_step(model, test_loader, criterion_test, device, epoch=epoch)
			best_test_acc = max(best_test_acc, test_acc)

		# Checkpoint
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
		with open(log_path, 'a') as af:
			train_cols = [run_id, epoch, 'train', train_loss, train_acc, best_train_acc]
			af.write('\t'.join([str(c) for c in train_cols]) + '\n')
			test_cols = [run_id, epoch, 'test', test_loss, test_acc, best_test_acc]
			af.write('\t'.join([str(c) for c in test_cols]) + '\n')

	return model


def soft_cross_entropy(pred, soft_targets, weights=None):
	if weights is not None:
		return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1) * weights, 1))
	else:
		return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


def train_step(model, train_loader, criterion, optimizer, epoch, device, log_interval=10):
	model.train()
	train_loss = 0.
	correct = 0
	total = 0
	train_loss_batch = 0
	epoch_size = len(train_loader.dataset)
	t_start = time.time()

	for batch_idx, (inputs, targets) in enumerate(train_loader):
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		if len(targets.size()) == 2:
			# Labels could be a posterior probability distribution. Use argmax as a proxy.
			target_probs, target_labels = targets.max(1)
		else:
			target_labels = targets
		correct += predicted.eq(target_labels).sum().item()

		prog = total / epoch_size
		exact_epoch = epoch + prog - 1
		acc = 100. * correct / total
		train_loss_batch = train_loss / total

		if (batch_idx + 1) % log_interval == 0:
			print('[Train] Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.1f} ({}/{})'.format(
				exact_epoch, batch_idx * len(inputs), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
				loss.item(), acc, correct, total))

	t_end = time.time()
	t_epoch = int(t_end - t_start)
	acc = 100. * correct / total

	return train_loss_batch, acc


def test_step(model, test_loader, criterion, device, epoch=0., silent=False):
	model.eval()
	test_loss = 0.
	correct = 0
	total = 0
	t_start = time.time()

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			nclasses = outputs.size(1)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	t_end = time.time()
	t_epoch = int(t_end - t_start)

	acc = 100. * correct / total
	test_loss /= total

	if not silent:
		print('[Test]  Epoch: {}\tLoss: {:.6f}\tAcc: {:.1f}% ({}/{})'.format(epoch, test_loss, acc,
																			 correct, total))

	return test_loss, acc
