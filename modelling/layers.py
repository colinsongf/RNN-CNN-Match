import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FullyConnected(nn.Module):

    def __init__(self,
                 sizes,
                 activation_function=torch.nn.ReLU(),
                 activation_function_output=None,
                 dense_layer_prefix='dense',
                 activation_function_prefix='activation'):
        """
        Simple FC layer. All you need is set sizes
        """

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

        # TODO as ModuleList
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
        """
        Simple RNN layer. All you need is set input_layer_size and hidden_size
        """

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
        """
        Pack padded sequence if set x_lengths if you have batch we variable length for better perfomance
        """

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


def gelu(x):
    """
    Gaussian Error Linear Unit implementation
    https://arxiv.org/pdf/1606.08415.pdf
    Used in transformer
    """

    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):

    def __init__(self):
        """
        Gaussian Error Linear Unit implementation
        https://arxiv.org/pdf/1606.08415.pdf
        Used in transformer
        """

        super(GELU, self).__init__()

        self.const = 0.044715

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class CNN(nn.Module):

    def __init__(self,
                 input_size,
                 out_chanels,
                 kernel_size_convolution,
                 kernel_size_pool=None,
                 pool_stride=1,
                 pool_layer=nn.MaxPool1d,
                 activation_function=GELU):
        """
        Simple CNN1D layer. All you need is set input_size, out_chanels and kernel_size_convolution
        """

        super(CNN, self).__init__()

        self.input_size = input_size
        self.out_chanels = out_chanels
        self.kernel_size_convolution = kernel_size_convolution

        self.convolution_layer = torch.nn.Conv1d(in_channels=self.input_size,
                                                 out_channels=self.out_chanels,
                                                 kernel_size=self.kernel_size_convolution)

        self.kernel_size_pool = kernel_size_pool if kernel_size_pool is not None else self.kernel_size_convolution
        self.activation_function = activation_function()
        self.pool_layer = pool_layer(kernel_size=self.kernel_size_pool, stride=pool_stride) \
            if pool_layer is not None else pool_layer

    def forward(self, x, x_lengths=None):
        """
        return correct batch with (batch_size x seq_len x input_size)  sizes
        """

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

        sim = 1 - (torch.acos(F.relu(self.cosine(u, v) - self.eps)) / math.pi)

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
