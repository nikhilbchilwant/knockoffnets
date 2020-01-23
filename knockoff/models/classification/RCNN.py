# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

__all__ = ['rcnn']

class RCNN(nn.Module):
	def __init__(self, num_classes, hidden_size, 
				 vocab_size, seq_len, embed_dim, weights=None):
		super(RCNN, self).__init__()
		
		"""
		Arguments
		---------
		num_classes : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embed_dim : Embedding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
		
		self.num_classes = num_classes
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embed_dim = embed_dim
		self.seq_len = seq_len
		
		self.word_embeddings = nn.Embedding(vocab_size, embed_dim)# Initializing the look-up table.
		if weights:
			self.word_embeddings.weight = \
				nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		self.dropout = 0.8
		self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True,
							dropout=self.dropout, bidirectional=True)
		self.W2 = nn.Linear(2*hidden_size+embed_dim, hidden_size)
		self.label = nn.Linear(hidden_size, num_classes)

		self._init_weights()


	def _init_weights(self):

		for m in self.parameters():

			if type(m) == type(nn.Linear):
				m.weight.data.uniform_(-.5, .5)
				m.bias.data.fill(0)

			elif type(m) == type(nn.Embedding):
				m.weight.data.uniform_(-.5, .5)

			elif type(m) == type(nn.LSTM):
				m.weight.data.uniform_(-.5, .5)
				m.bias.data.fill(0)
		
	def forward(self, input_sentence, input_lengths):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		input_length: The original length of each sequence

		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, num_classes)
		
		"""
		
		"""
		
		The idea of the paper "Recurrent Convolutional Neural Networks for Text Classification" is that we pass the embedding vector
		of the text sequences through a bidirectional LSTM and then for each sequence, our final embedding vector is the concatenation of 
		its own GloVe embedding and the left and right contextual embedding which in bidirectional LSTM is same as the corresponding hidden
		state. This final embedding is passed through a linear layer which maps this long concatenated encoding vector back to the hidden_size
		vector. After this step, we use a max pooling layer across all sequences of texts. This converts any varying length text into a fixed
		dimension tensor of size (batch_size, hidden_size) and finally we map this to the output layer.

		"""
		pad = torch.empty(input_sentence.size(1), self.seq_len-input_sentence.size(0), dtype=torch.int64).fill_(1).cuda()
		input_sentence = torch.cat((input_sentence.permute(1, 0), pad), 1)
		input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences, embed_dim)
		input = input.permute(1, 0, 2) 
		packed_input = pack_padded_sequence(input, input_lengths)

		output, (final_hidden_state, final_cell_state) = self.lstm(packed_input)
		output, _ = pad_packed_sequence(output, total_length=self.seq_len)
		
		final_encoding = torch.cat((output, input), 2).permute(1, 0, 2)
		y = self.W2(final_encoding) # y.size() = (batch_size, num_sequences, hidden_size)
		y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)
		y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_size, 1)
		y = y.squeeze(2)
		logits = self.label(y)
		
		return logits

def rcnn(**kwargs):

	return RCNN(**kwargs)
