# from torchtext.datasets import text_classification
#
# class data_preperator(text_classification):
#
# 	def __init__(self, vocab, data, labels):
# 		super(self, vocab, data, labels).__init__()
# 		self._data = data
# 		self._labels = labels
# 		self._vocab = vocab
#
# 	def CLOSED_WORLD(*args, **kwargs):
# 		""" Defines AG_NEWS datasets.
# 			The labels includes:
# 				- 1 : World
# 				- 2 : Sports
# 				- 3 : Business
# 				- 4 : Sci/Tech
#
# 		Create supervised learning dataset: AG_NEWS
#
# 		Separately returns the training and test dataset
#
# 		Arguments:
# 			root: Directory where the datasets are saved. Default: ".data"
# 			ngrams: a contiguous sequence of n items from s string text.
# 				Default: 1
# 			vocab: Vocabulary used for dataset. If None, it will generate a new
# 				vocabulary based on the train data set.
# 			include_unk: include unknown token in the data (Default: False)
#
# 		Examples:
# 			train_dataset, test_dataset = torchtext.datasets.AG_NEWS(ngrams=3)
#
# 		"""
#
# 		return _setup_datasets(*(("CLOSED_WORLD",) + args), **kwargs)
#
# 	DATASETS = {
# 		'CLOSED_WORLD': CLOSED_WORLD
# 	}
# 	LABELS = {
# 		'CLOSED_WORLD':{
# 			{1: 'World',
# 			 2: 'Sports',
# 			 3: 'Business',
# 			 4: 'Sci/Tech',
# 			 5: 'Health',
# 			 6: 'Entertainment',
# 			 7: 'Politics'}
# 		}
# 	}
#
# def _setup_datasets(dataset_name, root='.data', ngrams=2, vocab=None, include_unk=False):
# 	if (True):
# 		print("it works!")
# 	# dataset_tar = download_from_url(URLS[dataset_name], root=root)
# 	dataset_tar = 'closed_world.tar.gz'
# 	extracted_files = text_classification.extract_archive(dataset_tar)
#
# 	for fname in extracted_files:
# 		if fname.endswith('train.csv'):
# 			train_csv_path = fname
# 		if fname.endswith('test.csv'):
# 			test_csv_path = fname
# 		if fname.endswith('closed_world.csv'):
# 			closed_world_csv_path = fname
#
# 	if vocab is None:
# 		print('Building Vocab based on {}'.format(closed_world_csv_path))
# 		vocab = text_classification.build_vocab_from_iterator(text_classification._csv_iterator(closed_world_csv_path, ngrams))
# 	else:
# 		if not isinstance(vocab, text_classification.Vocab):
# 			raise TypeError("Passed vocabulary is not of type Vocab")
# 	print('Vocab has {} entries'.format(len(vocab)))
# 	print('Creating training data')
# 	train_data, train_labels = text_classification._create_data_from_iterator(
# 		vocab, text_classification._csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
# 	print('Creating testing data')
# 	test_data, test_labels = text_classification._create_data_from_iterator(
# 		vocab, text_classification._csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
# 	# if len(train_labels ^ test_labels) > 0:
# 	#     raise ValueError("Training and test labels don't match")
# 	return (text_classification.TextClassificationDataset(vocab, train_data, train_labels),
# 			text_classification.TextClassificationDataset(vocab, test_data, test_labels))
