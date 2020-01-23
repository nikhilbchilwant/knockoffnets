SELF_ATTENTION_CONFIG = ['num_classes', 'hidden_size', 'vocab_size', 'seq_len', 'embed_dim', 'weights']


def get_params(params, config):

	kwargs = {}

	for c in config:
		if c in params:
			kwargs[c] = params[c]

	return kwargs