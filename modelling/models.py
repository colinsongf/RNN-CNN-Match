import torch
import torch.nn as nn
from modelling.layers import NeuralNetwork, RNN, CNN
import torch.nn.functional as F
from modelling.templates import EmbeddingTemplate, TripletSimilarityTemplate


class DAN(nn.Module):

    def __init__(self,
                 sizes=(300, 256, 128, 100),
                 activation_function=torch.nn.ReLU(),
                 activation_function_output=None):

        super(DAN, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.neural_network = NeuralNetwork(sizes=sizes,
                                            activation_function=activation_function,
                                            activation_function_output=activation_function_output).to(self.device)

    def forward(self, x):

        x = x.mean(dim=1)

        x = self.neural_network(x)

        return x


class SimilarityDAN(TripletSimilarityTemplate):

    def __init__(self,
                 embedding_layer=None,
                 weight_file=None,
                 embedding_size=300,
                 sequence_max_length=32,
                 qa_model_same=False,
                 sizes=(300, 256, 128, 100),
                 activation_function=torch.nn.ReLU(),
                 activation_function_output=None,
                 delta=1):

        super(SimilarityDAN, self).__init__(question_model=DAN(sizes=sizes,
                                                               activation_function=activation_function,
                                                               activation_function_output=activation_function_output),

                                            answer_model=DAN(sizes=sizes,
                                                             activation_function=activation_function,
                                                             activation_function_output=activation_function_output),
                                            embedding_layer=embedding_layer,
                                            weight_file=weight_file,
                                            embedding_size=embedding_size,
                                            sequence_max_length=sequence_max_length,
                                            delta=delta)

        self.qa_model_same = qa_model_same

        if self.qa_model_same:
            self.answer_model = self.question_model

        self.question_model = self.question_model.to(self.device)
        self.answer_model = self.answer_model.to(self.device)


class RNNCNNMatch(nn.Module):

    def __init__(self,
                 embedding_size=300,
                 rnn_hidden_size=256,
                 cnn_hidden_size=128,
                 cnn_kernel_sizes=tuple(range(1, 6)),
                 kernel_size_pool=4):

        super(RNNCNNMatch, self).__init__()

        self.embedding_size = embedding_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_hidden_size = cnn_hidden_size
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.kernel_size_pool = kernel_size_pool

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.rnn = RNN(input_layer_size=self.embedding_size, hidden_size=self.rnn_hidden_size, output_last_state=False)

        self.cnns = [CNN(input_size=self.cnn_hidden_size, out_chanels=128, kernel_size_convolution=n, kernel_size_pool=4)
                     for n in self.cnn_kernel_sizes]

        self.cnns = [CNN(input_size=self.rnn_hidden_size,
                         out_chanels=self.cnn_hidden_size,
                         kernel_size_convolution=self.cnn_kernel_sizes[0],
                         kernel_size_pool=4)]

        if len(self.cnn_kernel_sizes) > 1:
            self.cnns.extend([CNN(input_size=self.cnn_hidden_size,
                                  out_chanels=self.cnn_hidden_size,
                                  kernel_size_convolution=kernel_size,
                                  kernel_size_pool=self.kernel_size_pool)
                              for kernel_size in self.cnn_kernel_sizes[1:]])

        self.layers = [self.rnn] + self.cnns

        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, sample):

        sample = self.model(sample)
        sample = sample.reshape(sample.size(0), 1, -1).squeeze()
        sample = F.softmax(sample)

        return sample
