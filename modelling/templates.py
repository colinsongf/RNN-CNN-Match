import torch
import torch.nn as nn
import torch.nn.functional as F
from modelling.layers import Embedding, USESimilarity, USETripletMarginLoss


# TODO implement some of this https://www.jeremyjordan.me/nn-learning-rate/ or use lr finder from fast.ai


class SimilarityTemplate(nn.Module):

    def __init__(self,
                 query_model,
                 candidate_model=None,
                 embedding_size=300,
                 embedding_weight_file=None,
                 embedding_layer_same=True,
                 sequence_max_length=32,
                 margin=1,
                 # similarity_function=USESimilarity,
                 similarity_function=torch.nn.CosineSimilarity,
                 loss_type='cross_entropy',
                 eps=1e-5,
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

        self.margin = margin
        self.similarity_function = similarity_function().to(self.device)

        self.loss_type = loss_type
        self.eps = eps

        if self.loss_type == 'cross_entropy':
            self.loss = nn.BCELoss()
        elif self.loss_type == 'triplet':
            self.loss = nn.TripletMarginLoss(margin=margin).to(self.device)
            # self.loss = USETripletMarginLoss(margin=margin)
        else:
            raise ValueError('Unknown loss type. Available: "cross_entropy" and "triplet"')

        self.loss = self.loss.to(self.device)

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

    def __compute_recall_cross_entropy__(self, query, candidate, target):

        similarity = self.similarity_function(query, candidate).round()

        return float((similarity == target).type(torch.FloatTensor).mean().cpu().numpy())

    def __compute_recall_triplet__(self, query, positive_candidate, negative_candidate):
        """
        Compute probability at which similarity_function(query, negative_candidate) is greater than
        similarity_function(queries, positive_candidate)
        """

        similarity_positive = self.similarity_function(query, positive_candidate)
        similarity_negative = self.similarity_function(query, negative_candidate)

        return float((similarity_positive > similarity_negative).type(torch.FloatTensor).mean().cpu().numpy())

    def compute_recall(self, *batch, vectorize=False):

        if vectorize:

            with torch.no_grad():

                if self.loss_type == 'cross_entropy':
                    target = batch[-1]
                    batch = self.forward(*batch[:-1])
                    batch += [target]
                elif self.loss_type == 'triplet':
                    batch = self.forward(*batch)

        if self.loss_type == 'cross_entropy':
            metric_function = self.__compute_recall_cross_entropy__
        elif self.loss_type == 'triplet':
            metric_function = self.__compute_recall_triplet__
        else:
            raise ValueError('Unknown loss_type')

        return metric_function(*batch)

    def compute_loss(self, *batch):

        if self.loss_type == 'cross_entropy':

            query, candidate = self.forward(query=batch[0], candidate=batch[1])

            target = torch.Tensor(batch[2]).to(self.device)

            similarity = self.similarity_function(query, candidate).clamp(min=self.eps, max=1-self.eps)

            vectorized_batch = [query, candidate, target]

            return self.loss(similarity, target), vectorized_batch

        elif self.loss_type == 'triplet':

            vectorized_batch = self.forward(*batch)

            return self.loss(*vectorized_batch), vectorized_batch

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
