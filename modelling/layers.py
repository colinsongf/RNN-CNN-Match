import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Embedding(nn.Module):

    def __init__(self,
                 embedding_size=300,
                 sequence_max_length=32,
                 pad_token='PAD',
                 pad_index=0,
                 pad_after=True,
                 embedding_layer=None,
                 verbose=False):

        super(Embedding, self).__init__()

        # TODO allennlp elmo embeddings

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_size = embedding_size
        self.output_size = self.embedding_size
        self.sequence_max_length = sequence_max_length

        self.pad_token = pad_token
        self.pad_index = pad_index

        self.pad_after = pad_after

        self.verbose = verbose

        self.token2index = {
            self.pad_token: self.pad_index
        }

        self.index2token = {
            self.pad_index: self.pad_token
        }

        self.weight_file = None
        self.embedding_layer = embedding_layer

        self.embeddings_type = None

    def __collect_pretrained_embeddings__(self, weight_file=None):

        self.weight_file = weight_file if weight_file is not None else self.weight_file

        if self.weight_file is None:
            raise ValueError('Need define weight file')

        embedding_matrix = [np.zeros(shape=(self.embedding_size,))]

        with open(file=self.weight_file, mode='r', encoding='utf-8', errors='ignore') as file:

            index = len(self.token2index)

            lines = tqdm(file.readlines(), desc='Collect embeddings') if self.verbose else file.readlines()

            for line in lines:

                line = line.split()

                word = ' '.join(line[:-self.embedding_size])
                embeddings = np.asarray(line[-self.embedding_size:], dtype='float32')

                if not word or embeddings.shape[0] != self.embedding_size:
                    continue

                self.token2index[word] = index
                self.index2token[index] = word

                embedding_matrix.append(embeddings)

                index += 1

        self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix)).to(self.device)

    def __create_embeddings_matrix__(self, vocab_size, token2index, index2token, pad_token, pad_index):

        self.token2index = token2index
        self.index2token = index2token
        self.pad_token = pad_token
        self.pad_index = pad_index

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=self.embedding_size,
                                            padding_idx=self.pad_index).to(self.device)

    def set_embeddings(self,
                       embeddings_type,
                       weight_file=None,
                       vocab_size=None,
                       token2index=None,
                       index2token=None,
                       pad_token=None,
                       pad_index=None):

        self.embeddings_type = embeddings_type

        if self.embeddings_type == 'pretrained':
            self.__collect_pretrained_embeddings__(weight_file=weight_file)
        elif self.embeddings_type == 'trainable':
            self.__create_embeddings_matrix__(vocab_size=vocab_size,
                                              token2index=token2index,
                                              index2token=index2token,
                                              pad_token=pad_token,
                                              pad_index=pad_index)
        else:
            raise ValueError('Unknown embeddings_type')

    def indexing(self, batch):

        sequence_max_length = self.sequence_max_length if self.sequence_max_length is not None \
            else max([len(sample) for sample in batch])

        for n_sample in range(len(batch)):

            tokens = batch[n_sample]

            if self.embeddings_type == 'pretrained':
                tokens = [self.token2index[token] for token in tokens if token in self.token2index]
                tokens = tokens[:sequence_max_length]

            if len(tokens) < sequence_max_length:

                pads = [self.pad_index] * (sequence_max_length - len(tokens))

                if self.pad_after:
                    tokens = tokens + pads
                else:
                    tokens = pads + tokens

                batch[n_sample] = tokens

        return torch.LongTensor(batch).to(self.device)

    def forward(self, batch):

        batch = self.indexing(batch=batch)

        return self.embedding_layer(batch).to(self.device)


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
                 pool_layer=nn.MaxPool1d,
                 activation_function=F.relu):

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

        self.activation_function = activation_function

    def forward(self, x, x_lengths=None):

        # Turn (batch_size x seq_len x input_size) into (batch_size x input_size x seq_len) for CNN
        x = x.transpose(1, 2)

        x = self.convolution_layer(x)

        x = self.activation_function(x)

        if self.pool_layer is not None:
            x = self.pool_layer(x)

        # Turn (batch_size x input_size x seq_len) into (batch_size x seq_len x input_size)
        x = x.transpose(1, 2)

        return x


class USESimilarity(nn.Module):

    """
    Similarity function from Universal Sentence Encoder:
    https://arxiv.org/pdf/1803.11175.pdf
    """

    def __init__(self, eps=1e-5):

        super(USESimilarity, self).__init__()

        self.eps = eps

        self.cosine = nn.CosineSimilarity()

    def forward(self, u, v):

        sim = 1 - (torch.acos(self.cosine(u, v) - self.eps) / math.pi)
        # sim = sim.clamp(min=self.eps, max=1-self.eps)

        return sim


class USETripletMarginLoss(nn.Module):

    """
    Loss class used Universal Sentence Encoder similarity function
    https://arxiv.org/pdf/1803.11175.pdf
    """

    def __init__(self, margin=1):

        super(USETripletMarginLoss, self).__init__()

        self.similarity_function = USESimilarity()
        self.margin = margin

    def forward(self, query, positive_candidate, negative_candidate):

        similarity_positive = self.similarity_function(query, positive_candidate)
        similarity_negative = self.similarity_function(query, negative_candidate)

        return F.relu(self.margin + similarity_positive - similarity_negative).mean()


def gelu(x):

    # TODO do as class

    """
    Gaussian Error Linear Unit implementation
    https://arxiv.org/pdf/1606.08415.pdf
    Used in transformer
    """

    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
