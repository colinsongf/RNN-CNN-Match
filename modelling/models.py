import torch
import torch.nn as nn
from modelling.layers import FullyConnected, RNN, CNN
import torch.nn.functional as F


class DAN(nn.Module):

    def __init__(self,
                 sizes=(300, 256, 128, 100),
                 activation_function=torch.nn.ReLU(),
                 activation_function_output=None):
        """
        Deep Average Network
        from
        Deep Unordered Composition Rivals Syntactic Methods for Text Classification
        https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf
        """

        super(DAN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.neural_network = FullyConnected(sizes=sizes,
                                             activation_function=activation_function,
                                             activation_function_output=activation_function_output).to(self.device)

    def forward(self, x):

        x = x.mean(dim=1)

        x = self.neural_network(x)

        return x


class RNNCNNMatch(nn.Module):

    def __init__(self,
                 embedding_size=300,
                 rnn_hidden_size=256,
                 cnn_hidden_size=128,
                 cnn_kernel_sizes=tuple(range(1, 6)),
                 kernel_size_pool=4):
        """
        RNN-CNN-Match
        from
        Neural Matching Models for Question Retrieval and Next Question Prediction in Conversation
        https://arxiv.org/pdf/1707.05409.pdf
        """

        super(RNNCNNMatch, self).__init__()

        self.embedding_size = embedding_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_hidden_size = cnn_hidden_size
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.kernel_size_pool = kernel_size_pool

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.layers = [RNN(input_layer_size=self.embedding_size,
                           hidden_size=self.rnn_hidden_size,
                           output_last_state=False)]

        self.layers += [CNN(input_size=self.rnn_hidden_size,
                            out_chanels=self.cnn_hidden_size,
                            kernel_size_convolution=self.cnn_kernel_sizes[0],
                            kernel_size_pool=self.kernel_size_pool)]

        if len(self.cnn_kernel_sizes) > 1:
            self.layers.extend([CNN(input_size=self.cnn_hidden_size,
                                    out_chanels=self.cnn_hidden_size,
                                    kernel_size_convolution=kernel_size,
                                    kernel_size_pool=self.kernel_size_pool)
                                for kernel_size in self.cnn_kernel_sizes[1:]])

        # TODO do params as hyper
        self.fully_connected = nn.Linear(in_features=896, out_features=300)

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, sample):

        sample = self.model(sample)
        sample = sample.reshape(sample.size(0), 1, -1).squeeze()

        sample = self.fully_connected(sample)
        sample = F.softmax(sample, dim=1)

        return sample
