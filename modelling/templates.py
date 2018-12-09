import torch
import torch.nn as nn
import torch.nn.functional as F
from modelling.layers import USESimilarity, USETripletMarginLoss


# TODO implement some of this https://www.jeremyjordan.me/nn-learning-rate/ or use lr finder from fast.ai


class SimilarityTemplate(nn.Module):

    def __init__(self,
                 query_model,
                 candidate_model=None,
                 vocab_size=80000,
                 embedding_size=300,
                 padding_idx=0,
                 embedding_matrix=None,
                 embedding_layer_same=True,
                 margin=1,
                 # similarity_function=USESimilarity,
                 similarity_function=torch.nn.CosineSimilarity,
                 loss_type='cross_entropy',
                 eps=1e-5):
        """
        Template for similarity models with 2 heads
        """

        super(SimilarityTemplate, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_size = embedding_size
        self.embedding_layer_same = embedding_layer_same

        self.query_embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size,
                                                        embedding_dim=embedding_size,
                                                        padding_idx=padding_idx).to(self.device)

        if self.embedding_layer_same:
            self.candidate_embedding_layer = self.query_embedding_layer
        else:
            self.candidate_embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size,
                                                                embedding_dim=embedding_size,
                                                                padding_idx=padding_idx).to(self.device)

        if embedding_matrix is not None:
            embedding_matrix = torch.Tensor(embedding_matrix).to(self.device)
            self.query_embedding_layer = self.query_embedding_layer.from_pretrained(embeddings=embedding_matrix)
            self.candidate_embedding_layer = self.query_embedding_layer

        self.query_model = query_model.to(self.device)
        self.candidate_model = candidate_model.to(self.device) if candidate_model is not None else self.query_model

        self.margin = margin
        self.similarity_function = similarity_function().to(self.device)

        self.loss_type = loss_type
        self.eps = eps

        if self.loss_type == 'cross_entropy':
            self.loss = nn.BCELoss()
        elif self.loss_type == 'triplet':
            self.loss = nn.TripletMarginLoss(margin=margin)
            # self.loss = USETripletMarginLoss(margin=margin)
        else:
            raise ValueError('Unknown loss type. Available: "cross_entropy" and "triplet"')

        self.loss = self.loss.to(self.device)

    def forward(self, query, candidate, negative_candidate_for_triplet=None):
        """
        For both: cross_entropy and triplet if you set correct loss function and negative_candidate_for_triplet
        """

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

    def __compute_recall_cross_entropy__(self, query, candidate, target, mean=True):
        """
        Compute recall for only cross-entropy
        """

        similarity = self.similarity_function(query, candidate).round()

        if mean:
            return float((similarity == target).type(torch.FloatTensor).mean().cpu().numpy())
        else:
            return [float(sample) for sample in (similarity == target).type(torch.FloatTensor).cpu().numpy()]

    def __compute_recall_triplet__(self, query, positive_candidate, negative_candidate, mean=True):
        """
        Compute probability at which similarity_function(query, negative_candidate) is greater than
        similarity_function(queries, positive_candidate)
        """

        similarity_positive = self.similarity_function(query, positive_candidate)
        similarity_negative = self.similarity_function(query, negative_candidate)

        if mean:
            return float((similarity_positive > similarity_negative).type(torch.FloatTensor).mean().cpu().numpy())
        else:
            return [float(sample) for sample in
                    (similarity_positive > similarity_negative).type(torch.FloatTensor).cpu().numpy()]

    def compute_recall(self, *batch, vectorize=False, mean=True):
        """
        Compute recall for both with auto choose
        """

        if vectorize:

            with torch.no_grad():

                if self.loss_type == 'cross_entropy':
                    target = batch[-1]
                    batch = self.forward(*batch[:-1])
                    batch += [target]
                elif self.loss_type == 'triplet':
                    batch = self.forward(*batch)

        if self.loss_type == 'cross_entropy':
            return self.__compute_recall_cross_entropy__(*batch, mean=mean)
        elif self.loss_type == 'triplet':
            return self.__compute_recall_triplet__(*batch, mean=mean)
        else:
            raise ValueError('Unknown loss_type')

    def compute_loss(self, *batch):
        """
        Compute loss for both with auto choose
        """

        if self.loss_type == 'cross_entropy':

            query, candidate = self.forward(query=batch[0], candidate=batch[1])

            target = batch[2]

            # solve Assertion `x >= 0. && x <= 1.' failed. input value should be between 0~1, but got 1.000000
            # occurs at the beginning of training because softmax
            similarity = F.relu(self.similarity_function(query, candidate) - self.eps)

            vectorized_batch = [query, candidate, target]

            return self.loss(similarity, target), vectorized_batch

        elif self.loss_type == 'triplet':

            vectorized_batch = self.forward(*batch)

            return self.loss(*vectorized_batch), vectorized_batch

    def text_embedding(self, x, model_type='query'):
        """
        Text embedding with no gradients
        """

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
