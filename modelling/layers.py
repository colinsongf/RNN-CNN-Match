import numpy as np
from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):

    # TODO trainable embeddings

    def __init__(self):

        super(Embedding, self).__init__()

    def forward(self, x):

        return x


class EmbeddingFromPretrained(nn.Module):

    def __init__(self,
                 weight_file,
                 vector_size,
                 sequence_max_length=32,
                 pad_token='PAD',
                 pad_after=True,
                 existing_words=None,
                 verbose=False):

        super(EmbeddingFromPretrained, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.weight_file = weight_file
        self.vector_size = vector_size
        self.output_size = self.vector_size
        self.sequence_max_length = sequence_max_length

        self.pad_token = pad_token
        self.pad_index = 0

        self.pad_after = pad_after

        self.existing_words = existing_words

        self.word2index = {
            self.pad_token: self.pad_index
        }

        self.index2word = {
            self.pad_index: self.pad_token
        }

        self.embedding_layer = self.__collect_embeddings__(verbose=verbose)

    def __collect_embeddings__(self, verbose=False):

        embedding_matrix = [np.zeros(shape=(self.vector_size, ))]

        with open(file=self.weight_file, mode='r', encoding='utf-8', errors='ignore') as file:

            index = len(self.word2index)

            lines = tqdm(file.readlines(), desc='Collect embeddings') if verbose else file.readlines()

            for line in lines:

                line = line.split()

                word = ' '.join(line[:-self.vector_size])
                embeddings = np.asarray(line[-self.vector_size:], dtype='float32')

                if not word or embeddings.shape[0] != self.vector_size or \
                        (self.existing_words is not None and word not in self.existing_words):
                    continue

                self.word2index[word] = index
                self.index2word[index] = word

                embedding_matrix.append(embeddings)

                index += 1

        return torch.nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix)).to(self.device)

    def forward(self, input_batch, targets_batch=None, permutation=False):

        sequence_max_length = self.sequence_max_length if self.sequence_max_length is not None \
            else max([len(sample) for sample in input_batch])

        sequence_lengths = []

        embedded_batch = torch.Tensor(size=(len(input_batch), sequence_max_length, self.vector_size)).to(self.device)

        for n_sample in range(len(input_batch)):

            tokens = [self.word2index[token] for token in input_batch[n_sample] if token in self.word2index]
            tokens = tokens[:sequence_max_length]

            if not tokens:
                if targets_batch is not None:
                    targets_batch.pop(n_sample)
                continue

            sequence_lengths.append(len(tokens))

            if len(tokens) < sequence_max_length:

                pads = [self.pad_index] * (sequence_max_length - len(tokens))

                if self.pad_after:
                    tokens = tokens + pads
                else:
                    tokens = pads + tokens

            tokens = torch.LongTensor(tokens).to(self.device)

            embedded_batch[n_sample] = self.embedding_layer(tokens).to(self.device)

        if targets_batch is not None:
            targets_batch = torch.Tensor(targets_batch).to(self.device)

        if embedded_batch.sum() == 0:
            return None, None, None
        elif not permutation:
            return embedded_batch

        sequence_lengths = torch.Tensor(sequence_lengths)

        sequence_lengths, permutation_idx = sequence_lengths.sort(descending=True)

        embedded_batch = embedded_batch[permutation_idx]
        sequence_lengths = sequence_lengths.to(self.device)

        if targets_batch is not None:
            targets_batch = targets_batch[permutation_idx]
            return embedded_batch, sequence_lengths, targets_batch
        else:
            return embedded_batch, sequence_lengths, permutation_idx


class FullyConnected(nn.Module):

    def __init__(self,
                 sizes,
                 activation_function=torch.nn.ReLU(),
                 activation_function_output=None,
                 dense_layer_prefix='dense',
                 activation_function_prefix='activation'):

        super(FullyConnected, self).__init__()

        self.sizes = list(sizes)
        self.activation_function = activation_function
        self.activation_function_output = activation_function_output if activation_function_output \
                                                                        is not None else self.activation_function

        self.dense_layer_prefix = dense_layer_prefix
        self.activation_function_prefix = activation_function_prefix

        self.input_size = self.sizes[0]
        self.output_size = self.sizes[-1]

        self.layers = []

        for n in range(len(self.sizes[:-1])):

            self.__dict__['{}_{}'.format(self.dense_layer_prefix, n)] = nn.Linear(in_features=self.sizes[n],
                                                                                  out_features=self.sizes[n+1])

            if n == len(self.sizes) - 2:
                current_activation_function = self.activation_function_output

            else:
                current_activation_function = self.activation_function

            self.__dict__['{}_{}'.format(self.activation_function_prefix, n)] = current_activation_function

            self.layers.append(self.__dict__['{}_{}'.format(self.dense_layer_prefix, n)])
            self.layers.append(self.__dict__['{}_{}'.format(self.activation_function_prefix, n)])

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x, x_lengths=None):

        x = self.model(x)

        return x


class RNN(nn.Module):

    def __init__(self,
                 input_layer_size,
                 hidden_size=256,
                 rnn_type=nn.LSTM,
                 rnn_layers=1,
                 dropout=0.,
                 bidirectional=False,
                 output_last_state=True):

        super(RNN, self).__init__()

        self.input_size = input_layer_size
        self.output_last_state = output_last_state
        self.hidden_size = hidden_size

        self.rnn_type = rnn_type
        self.rnn_layers = rnn_layers

        self.dropout = dropout
        self.bidirectional = bidirectional

        self.output_size = self.hidden_size

        if self.bidirectional:
            self.output_size *= 2

        self.hidden_states = None
        self.cell_states = None

        self.rnn = self.rnn_type(input_size=self.input_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.rnn_layers,
                                 dropout=self.dropout if self.rnn_layers > 1 else 0.,
                                 bidirectional=self.bidirectional)

        self.last_hidden_size = self.hidden_size

    def forward(self, x, x_lengths=None):

        x = x.transpose(0, 1)  # (B,L,D) -> (L,B,D)

        if x_lengths is not None:
            x = pack_padded_sequence(x, x_lengths)

        x, internal_rnn_data = self.rnn(x)

        if type(internal_rnn_data) == tuple:
            self.hidden_states, self.cell_states = internal_rnn_data
        else:
            self.hidden_states = internal_rnn_data

        if x_lengths is not None:
            rnn_output, _ = pad_packed_sequence(x)
        else:
            rnn_output = x

        rnn_output = rnn_output.transpose(0, 1)  # (L,B,D) -> (B,L,D)

        if self.output_last_state:
            return rnn_output[:, -1, :]
        else:
            if x_lengths is not None:
                return rnn_output, x_lengths
            else:
                return rnn_output


class CNN(nn.Module):

    def __init__(self,
                 input_size,
                 out_chanels,
                 kernel_size_convolution,
                 kernel_size_pool=None,
                 pool_layer=nn.MaxPool1d):

        super(CNN, self).__init__()

        self.input_size = input_size
        self.out_chanels = out_chanels
        self.kernel_size_convolution = kernel_size_convolution

        self.convolution_layer = torch.nn.Conv1d(in_channels=self.input_size,
                                                 out_channels=self.out_chanels,
                                                 kernel_size=self.kernel_size_convolution)

        self.kernel_size_pool = kernel_size_pool if kernel_size_pool is not None else self.kernel_size_convolution
        self.pool_layer = pool_layer(kernel_size=self.kernel_size_pool, stride=1) \
            if pool_layer is not None else pool_layer

    def forward(self, x, x_lengths=None):

        # Turn (batch_size x seq_len x input_size) into (batch_size x input_size x seq_len) for CNN
        x = x.transpose(1, 2)

        x = self.convolution_layer(x)

        if self.pool_layer is not None:
            x = self.pool_layer(x)

        # Turn (batch_size x input_size x seq_len) into (batch_size x seq_len x input_size)
        x = x.transpose(1, 2)

        return x


class USESimilarity(nn.Module):

    """
    Distance function from Universal Sentence Encoder:
    https://arxiv.org/pdf/1803.11175.pdf
    """

    def __init__(self, eps=1e-6):

        super(USESimilarity, self).__init__()

        self.eps = eps

        self.cosine = nn.CosineSimilarity()

    def forward(self, u, v):

        sim = 1 - (torch.acos(self.cosine(u, v) - self.eps) / math.pi)

        return sim
