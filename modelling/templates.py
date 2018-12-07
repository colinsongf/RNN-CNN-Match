import torch
import torch.nn as nn
import torch.nn.functional as F
from modelling.layers import Embedding, EmbeddingFromPretrained, USESimilarity


class EmbeddingTemplate(nn.Module):

    def __init__(self,
                 embedding_layer=None,
                 weight_file=None,
                 embedding_size=300,
                 sequence_max_length=32,
                 verbose=False):

        super(EmbeddingTemplate, self).__init__()

        if embedding_layer is not None:
            self.embedding_layer = embedding_layer
        elif weight_file is not None:
            self.embedding_layer = EmbeddingFromPretrained(weight_file=weight_file,
                                                           vector_size=embedding_size,
                                                           sequence_max_length=sequence_max_length,
                                                           verbose=verbose)
        elif embedding_size is not None:
            self.embedding_layer = Embedding()
        else:
            raise ValueError('Need define embedding layer, weight file or embedding size')

        self.embedding_layer = self.embedding_layer.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, *inputs):

        return self.embedding_layer(*inputs)


class TripletSimilarityTemplate(EmbeddingTemplate):

    def __init__(self,
                 query_model,
                 candidate_model=None,
                 embedding_layer=None,
                 weight_file=None,
                 embedding_size=300,
                 sequence_max_length=32,
                 delta=1,
                 similarity_function=torch.nn.CosineSimilarity()):

        super(TripletSimilarityTemplate, self).__init__(embedding_layer=embedding_layer,
                                                        weight_file=weight_file,
                                                        embedding_size=embedding_size,
                                                        sequence_max_length=sequence_max_length)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.query_model = query_model.to(self.device)
        self.candidate_model = candidate_model.to(self.device) if candidate_model is not None else self.query_model

        self.delta = delta
        self.similarity_function = similarity_function

    def forward(self, query, positive_candidate, negative_candidate, need_loss=False):

        inputs = [query, positive_candidate, negative_candidate]

        for n_input in range(len(inputs)):
            inputs[n_input] = self.embedding_layer(inputs[n_input])

        inputs[0] = self.query_model(inputs[0])

        for n_input in range(1, len(inputs)):
            inputs[n_input] = self.candidate_model(inputs[n_input])

        if need_loss:
            inputs.append(self.compute_triplet_loss(*inputs))

        return inputs

    def compute_triplet_loss(self, query, positive_candidate, negative_candidate, vectorize=False):

        if vectorize:
            query, positive_candidate, negative_candidate = self.forward(query, positive_candidate, negative_candidate)

        similarity_positive = self.similarity_function(query, positive_candidate)
        similarity_negative = self.similarity_function(query, negative_candidate)

        return F.relu(self.delta + similarity_positive - similarity_negative).mean()

    def sentence_embedding(self, x, model_type='query'):

        if model_type == 'query':
            model = self.query_model
        elif model_type == 'candidate':
            model = self.candidate_model
        else:
            raise ValueError('Unknown model_type')

        with torch.no_grad():
            x = self.embedding_layer(x)
            x = model(x)

        return x


class SimilarityTemplate(nn.Module):

    def __init__(self,
                 query_model,
                 candidate_model=None,
                 embedding_layer=None,
                 weight_file=None,
                 embedding_size=300,
                 sequence_max_length=32,
                 embedding_layer_same=True,
                 delta=1,
                 similarity_function=USESimilarity,
                 loss=nn.BCELoss,
                 verbose=False):

        super(SimilarityTemplate, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_layer_same = embedding_layer_same

        self.query_embedding_layer = EmbeddingTemplate(embedding_layer=embedding_layer,
                                                       weight_file=weight_file,
                                                       embedding_size=embedding_size,
                                                       sequence_max_length=sequence_max_length,
                                                       verbose=verbose).to(self.device)

        if weight_file is not None or self.embedding_layer_same:
            self.candidate_embedding_layer = self.query_embedding_layer
        else:
            self.candidate_embedding_layer = EmbeddingTemplate(embedding_layer=embedding_layer).to(self.device)

        self.query_model = query_model.to(self.device)
        self.candidate_model = candidate_model.to(self.device) if candidate_model is not None else self.query_model

        self.delta = delta
        self.similarity_function = similarity_function().to(self.device)

        self.loss = loss().to(self.device)

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

    def compute_triplet_loss(self, query, positive_candidate, negative_candidate, vectorize=False):

        if vectorize:
            query, positive_candidate, negative_candidate = self.forward(query, positive_candidate, negative_candidate)

        similarity_positive = self.similarity_function(query, positive_candidate)
        similarity_negative = self.similarity_function(query, negative_candidate)

        return F.relu(self.delta + similarity_positive - similarity_negative).mean()

    def compute_cross_entropy(self, query, candidate, target):

        similarity = self.similarity_function(query, candidate)

        if self.loss is not None:
            return self.loss(similarity, target)
        else:
            raise ValueError('Need define cross_entropy loss')

    def sentence_embedding(self, x, model_type='query'):

        if model_type == 'query':
            model = self.query_model
        elif model_type == 'candidate':
            model = self.candidate_model
        else:
            raise ValueError('Unknown model_type')

        with torch.no_grad():
            x = self.embedding_layer(x)
            x = model(x)

        return x
