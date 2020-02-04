# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

__all__ = ['attention_model']

class AttentionModel(torch.nn.Module):
	def __init__(self, num_classes, hidden_size,  
				 vocab_size, seq_len, embed_dim, 
				 num_layers=1, dropout=.5, weights=None):
		super(AttentionModel, self).__init__()
		
		"""
		Arguments
		---------
		num_classes : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embed_dim : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		--------
		
		"""
		
		self.num_classes = num_classes
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.seq_len = seq_len
		self.embed_dim = embed_dim
		
		self.word_embeddings = nn.Embedding(vocab_size, embed_dim)

		if weights is not None:
			self.word_embeddings.load_state_dict({'weight': weights})
			self.word_embeddings.weight.requires_grad = False
		else:
			self.word_embeddings.weight.requires_grad = True
				
		self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, 
							num_layers=num_layers,  dropout=dropout)
		self.label = nn.Linear(hidden_size*num_layers, num_classes)

		self._init_weights()
		
	def attention_net(self, lstm_output, final_state):

		""" 
		Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
		between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
		
		Arguments
		---------
		
		lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
		final_state : Final time-step hidden state (h_n) of the LSTM
		
		---------
		
		Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
				  new hidden state.
				  
		Tensor Size :
					hidden.size() = (batch_size, hidden_size)
					attn_weights.size() = (batch_size, num_seq)
					soft_attn_weights.size() = (batch_size, num_seq)
					new_hidden_state.size() = (batch_size, hidden_size)
					  
		"""
		
		# hidden = final_state.squeeze(0)
		hidden = final_state.permute(1, 2, 0)
		lstm_output = lstm_output.permute(1, 0, 2)
		attn_weights = torch.bmm(lstm_output, hidden)
		# Alt Code!
		# attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(
			lstm_output.transpose(1, 2), soft_attn_weights)

		# Alt Code!
		# new_hidden_state = torch.bmm(
		# 	lstm_output.transpose(1, 2), 
		# 	soft_attn_weights.unsqueeze(2)).squeeze(2)
		
		return new_hidden_state.view(new_hidden_state.size(0), -1)


	def _init_weights(self):

		for m in self.parameters():

			if type(m) == type(nn.Linear):
				m.weight.data.uniform_(-.5, .5)
				m.bias.data.fill(0)

			elif type(m) == type(nn.Embedding):
				if m.weight.requires_grad:
					m.weight.data.uniform_(-.5, .5)

			elif type(m) == type(nn.LSTM):
				m.weight.data.uniform_(-.5, .5)
				m.bias.data.fill(0)				

	
	def forward(self, input_sentences, input_lengths):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		input_length: The original length of each sequence

		Returns
		-------
		Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
		final_output.shape = (batch_size, num_classes)
		
		"""
		
		input = self.word_embeddings(input_sentences)
		input = pack_padded_sequence(input, input_lengths)
		# input = input.permute(1, 0, 2)
		
		# if batch_size is None:
		# 	h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
		# 	c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
		# else:
		# 	h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		# 	c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		
		output, (final_hidden_state, final_cell_state) = self.lstm(input)
		output, _ = pad_packed_sequence(output, total_length=self.seq_len)
		# print('output.size():', output.size())
		# print('final_hidden_state.size():', final_hidden_state.size())
		# output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) # final_hidden_state.size() = (1, batch_size, hidden_size) 
		# output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
		
		attn_output = self.attention_net(output, final_hidden_state)
		logits = self.label(attn_output)
		
		return logits

def attention_model(**kwargs):

	return AttentionModel(**kwargs)
