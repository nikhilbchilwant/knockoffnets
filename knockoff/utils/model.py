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
