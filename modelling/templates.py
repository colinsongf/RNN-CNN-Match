import torch
import torch.nn as nn
from modelling.layers import Embedding, USESimilarity


class SimilarityTemplate(nn.Module):

    def __init__(self,
                 query_model,
                 candidate_model=None,
                 embedding_size=300,
                 embedding_weight_file=None,
                 embedding_layer_same=True,
                 sequence_max_length=32,
                 delta=1,
                 similarity_function=USESimilarity,
                 loss_type='cross_entropy',
                 verbose=False):

        super(SimilarityTemplate, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_size = embedding_size
        self.embedding_weight_file = embedding_weight_file
        self.embedding_layer_same = embedding_layer_same
        self.sequence_max_length = sequence_max_length
        self.verbose = verbose

        self.query_embedding_layer = Embedding(embedding_size=self.embedding_size,
                                               sequence_max_length=self.sequence_max_length,
                                               verbose=self.verbose).to(self.device)

        if self.embedding_weight_file is not None or self.embedding_layer_same:
            self.candidate_embedding_layer = self.query_embedding_layer
        else:
            self.candidate_embedding_layer = Embedding(embedding_size=self.embedding_size,
                                                       sequence_max_length=self.sequence_max_length,
                                                       verbose=self.verbose).to(self.device)

        self.query_model = query_model.to(self.device)
        self.candidate_model = candidate_model.to(self.device) if candidate_model is not None else self.query_model

        self.delta = delta
        self.similarity_function = similarity_function().to(self.device)

        self.loss_type = loss_type

        if self.loss_type == 'cross_entropy':
            self.loss = nn.BCELoss().to(self.device)
        elif self.loss_type == 'triplet':
            self.loss = nn.TripletMarginLoss(margin=delta).to(self.device)
        else:
            raise ValueError('Unknown loss type. Available: "cross_entropy" and "triplet"')

    def forward(self, query, candidate, negative_candidate_for_triplet=None):

        inputs = [query, candidate]

        if negative_candidate_for_triplet is not None:
            inputs.append(negative_candidate_for_triplet)

        inputs[0] = self.query_embedding_layer(inputs[0])

        for n_input in range(1, len(inputs)):
            inputs[n_input] = self.candidate_embedding_layer(inputs[n_input])

        inputs[0] = self.query_model(inputs[0])

        for n_input in range(1, len(inputs)):
            inputs[n_input] = self.candidate_model(inputs[n_input])

        return inputs

    def compute_cross_entropy(self, query, candidate, target):

        similarity = self.similarity_function(query, candidate)

        if self.loss is not None:
            return self.loss(similarity, target)
        else:
            raise ValueError('Need define cross_entropy loss')

    def compute_loss(self, *batch):

        if self.loss_type == 'cross_entropy':

            query, candidate = self.forward(query=batch[0], candidate=batch[1])

            target = torch.Tensor(batch[2])

            batch = [query, candidate, target]

            return self.compute_cross_entropy(*batch)
        elif self.loss_type == 'triplet':

            batch = self.forward(*batch)

            return self.loss(*batch)

    def text_embedding(self, x, model_type='query'):

        if model_type == 'query':
            model = self.query_model
            embedder = self.query_embedding_layer
        elif model_type == 'candidate':
            model = self.candidate_model
            embedder = self.candidate_embedding_layer
        else:
            raise ValueError('Unknown model_type')

        with torch.no_grad():
            x = embedder(x)
            x = model(x)

        return x
