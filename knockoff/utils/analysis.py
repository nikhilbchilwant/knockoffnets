import json
import os.path as osp
from functools import partial

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

def get_datasets_from_param(param_folder):

	params_path = osp.join(param_folder, 'params.json')
	with open(params_path) as jf:
		params = json.load(jf)

	return params['dataset']


def count_total_unk_percent_wrt_victim(adversary_dataset, collate_fn):

	unk_count = 0 # Count for the unk words
	total_num = 0
	unk_percentage = .0
	pbar = tqdm(total=len(adversary_dataset))
	data = DataLoader(adversary_dataset, batch_size=1024, collate_fn=collate_fn, num_workers=10)

	for i, (text, _, _) in enumerate(data):
		num_data = sum([t.size(0) for t in text])
		unk_count += torch.sum(text == 0, dtype=torch.float).item()
		total_num += num_data
		pbar.update(len(_))
		
	print('total num:{}'.format(total_num))
	return unk_count / total_num

