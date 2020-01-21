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

	model_out_path = osp.join(out_path, 
		'checkpoint-{}-{}.pth.tar'.format(model_name, modelfamily))

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
