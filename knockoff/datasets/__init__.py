dataset_to_modelfamily = {

	### TextClassificationDataset
	'AG_NEWS': 'classification',
	'SogouNews': 'classification',
	'DBpedia': 'classification',
	'YelpReviewPolarity': 'classification',
	'YelpReviewFull': 'classification',
	'YahooAnswers': 'classification',
	'AmazonReviewPolarity': 'classification',
	'AmazonReviewFull': 'classification',
	'EnWik9': 'classification',
	'TREC': 'classification',

	### LanguageModelingDataset'
	'SNLI': 'entailment',
	'MultiNLI': 'entailment',
	'XNLI': 'entailment',

	###Sentiment classification
	'SST': 'sentiment',
	'IMDB': 'sentiment',
}

dataset_metadata = {
	'AG_NEWS': {
		'ngram': 2
	},
	'default': {
		'ngram': 2
	}
}
