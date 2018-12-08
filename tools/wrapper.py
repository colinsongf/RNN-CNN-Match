import numpy as np
import torch
from tqdm import tqdm
import random
try:
    import telegram_send
except ModuleNotFoundError:
    pass
from matplotlib import pyplot as plt


class Wrapper:

    def __init__(self,
                 dataset,
                 model,
                 optimizer,
                 model_name='default_model',
                 batch_size=32,
                 cross_entropy_negative_k=None,
                 generate_negatives_type='random',
                 hard_negatives_multiplier=3,
                 hard_k_next=False,
                 embeddings_type='pretrained',
                 embeddings_weight_file=None):

        super(Wrapper, self).__init__()

        self.available_embeddings_types = ['trainable', 'pretrained']
        self.available_generate_negatives_types = ['random', 'hard']

        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.model_name = model_name
        self.batch_size = batch_size
        self.cross_entropy_negative_k = cross_entropy_negative_k if cross_entropy_negative_k is not None \
            else self.batch_size
        self.embeddings_type = embeddings_type
        self.embeddings_weight_file = embeddings_weight_file
        self.generate_negatives_type = generate_negatives_type
        self.hard_negatives_multiplier = hard_negatives_multiplier
        self.hard_k_next = hard_k_next

        if self.embeddings_type not in self.available_embeddings_types:
            raise ValueError('Unknown embeddings_type. Available: {}'.format(', '.join(
                self.available_embeddings_types)))

        if self.generate_negatives_type not in self.available_generate_negatives_types:
            raise ValueError('Unknown generate_negatives_type. Available: {}'.format(', '.join(
                self.available_generate_negatives_types)))

        self.losses = []
        self.recalls = []
        self.epoch_mean_losses = []
        self.epoch_mean_recalls = []
        self.epochs_passed = 0

        self.validation_losses = []
        self.validation_recalls = []

    def collect_data(self, embeddings_weight_file=None):

        self.embeddings_weight_file = embeddings_weight_file if embeddings_weight_file is not None \
            else self.embeddings_weight_file

        self.dataset.collect()

        if self.embeddings_type == 'trainable':

            embedding_layers_params = {
                'embeddings_type': self.embeddings_type,
                'vocab_size': len(self.dataset),
                'token2index': self.dataset.vocabulary.get_tokens2index,
                'index2token': self.dataset.vocabulary.index2token,
                'pad_token': self.dataset.vocabulary.pad_token,
                'pad_index': self.dataset.vocabulary[self.dataset.vocabulary.pad_token]
            }

        else:

            if self.embeddings_weight_file is None:
                raise ValueError('Need define embeddings_weight_file')

            embedding_layers_params = {
                'embeddings_type': self.embeddings_type,
                'weight_file': self.embeddings_weight_file
            }

        self.model.query_embedding_layer.set_embeddings(**embedding_layers_params)

        if not self.model.embedding_layer_same:
            self.model.candidate_embedding_layer.set_embeddings(**embedding_layers_params)

    def __get_samples__(self, data, start, stop):

        tmp_data = data[start:stop]

        queries = self.dataset.qids2questions(tmp_data.qid1)
        positive_candidates = self.dataset.qids2questions(tmp_data.qid2)

        return queries, positive_candidates

    def get_random_negatives(self, samples=None):

        samples = samples if samples is not None else self.batch_size

        random_qids = random.sample(self.dataset.qid2question.keys(), samples)

        return self.dataset.qids2questions(batch=random_qids)

    def get_hard_negatives(self, queries, samples):

        samples = int(samples)

        samples = min(samples, len(self.dataset.validation))

        negatives = self.get_random_negatives(samples=samples)

        query_vactorized = self.model.text_embedding(x=queries)

        negatives_vectorized = self.model.text_embedding(x=negatives, model_type='candidate')
        attention = torch.matmul(query_vactorized, negatives_vectorized.transpose(0, 1))

        if self.hard_k_next or samples == len(self.dataset.validation):

            # if we wont get top-2
            # if we take too many samples we can choose select sample existing in quieries

            max_attentive_matrix = torch.zeros_like(attention)
            values, args = attention.max(dim=1)

            for n, i in enumerate(range(max_attentive_matrix.size(0))):
                max_attentive_matrix[i, args[n]] = values[n]

            attention -= max_attentive_matrix

        max_attentive_indexes = list(attention.argmax(dim=1).cpu().numpy())

        max_attentive = [negatives[n] for n in max_attentive_indexes]

        return max_attentive

    def __generate_negatives__(self, queries, batch_size):

        if self.generate_negatives_type == 'random':
            return self.get_random_negatives(samples=batch_size)
        else:
            return self.get_hard_negatives(queries=queries, samples=batch_size*self.hard_negatives_multiplier)

    def __cross_entropy_batch_generator__(self, data, batch_size):

        positives_batch_size = batch_size - self.cross_entropy_negative_k

        for n_batch in range(round(len(data) / batch_size)):

            queries, positive_candidates = self.__get_samples__(data=data,
                                                                start=n_batch*positives_batch_size,
                                                                stop=(n_batch+1)*positives_batch_size)

            negative_candidates = self.__generate_negatives__(queries=queries,
                                                              batch_size=self.cross_entropy_negative_k)

            targets = [1 for _ in range(len(queries))] + [0 for _ in range(len(queries))]

            queries *= 2
            candidates = positive_candidates + negative_candidates

            indexes = list(range(len(queries)))
            random.shuffle(indexes)

            queries = [queries[index] for index in indexes]
            candidates = [candidates[index] for index in indexes]
            targets = [targets[index] for index in indexes]

            yield queries, candidates, targets

    def __triplet_batch_generator__(self, data, batch_size):

        for n_batch in range(round(len(data) / batch_size)):

            queries, positive_candidates = self.__get_samples__(data=data,
                                                                start=n_batch*batch_size,
                                                                stop=(n_batch+1)*batch_size)

            negative_candidates = self.__generate_negatives__(queries=queries,
                                                              batch_size=batch_size)

            indexes = list(range(len(queries)))
            random.shuffle(indexes)

            queries = [queries[index] for index in indexes]
            positive_candidates = [positive_candidates[index] for index in indexes]
            negative_candidates = [negative_candidates[index] for index in indexes]

            yield queries, positive_candidates, negative_candidates

    def batch_generator(self, data_type='train', batch_size=None):

        data = self.dataset.__dict__[data_type]
        batch_size = batch_size if batch_size is not None else self.batch_size

        if self.model.loss_type == 'cross_entropy':
            return self.__cross_entropy_batch_generator__(data=data, batch_size=batch_size)
        elif self.model.loss_type == 'triplet':
            return self.__triplet_batch_generator__(data=data, batch_size=batch_size)

    def compute_loss_recall(self, batch):

        loss, vectorized_batch = self.model.compute_loss(*batch)
        recall = self.model.compute_recall(*vectorized_batch)

        return loss, recall

    def train(self,
              epochs=5,
              negatives_type=None,
              verbose=False):

        self.generate_negatives_type = negatives_type if negatives_type is not None else self.generate_negatives_type

        for n_epoch in range(1, epochs+1):

            if verbose:

                total_n_batches = len(self.dataset.train) // self.batch_size

                if self.model.loss_type == 'cross_entropy':
                    total_n_batches += self.cross_entropy_negative_k * total_n_batches

                pbar = tqdm(total=total_n_batches, desc='Train Epoch {}'.format(n_epoch))

            batch_losses = []
            batch_recalls = []

            for batch in self.batch_generator(data_type='train'):

                # loss, vectorized_batch = self.model.compute_loss(*batch)
                # recall = self.model.compute_recall(*vectorized_batch)

                loss, recall = self.compute_loss_recall(batch)

                batch_recalls.append(recall)
                self.recalls.append(recall)

                self.losses.append(loss.item())
                batch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if verbose:
                    pbar.update(1)

            self.epochs_passed += 1

            self.epoch_mean_losses.append(np.mean(batch_losses))
            self.epoch_mean_recalls.append(np.mean(batch_recalls))

            for batch in self.batch_generator(data_type='validation', batch_size=len(self.dataset.validation)):

                with torch.no_grad():

                    # validation_loss, vectorized_batch = self.model.compute_loss(*batch)
                    # validation_recall = self.model.compute_recall(*vectorized_batch)

                    validation_loss, validation_recall = self.compute_loss_recall(batch)

                    self.validation_losses.append(validation_loss)
                    self.validation_recalls.append(validation_recall)

            if verbose:
                pbar.close()

                message = list()

                message.append(
                    'Epoch: [{}/{}] | {} loss: {:.3f} | Validation Loss: {:.3f}'.format(
                        n_epoch + self.epochs_passed,
                        epochs + self.epochs_passed,
                        self.model.loss_type.capitalize(),
                        self.epoch_mean_losses[-1],
                        self.validation_losses[-1]
                    )
                )

                message.append(
                    'Mean Recall: {:.2f} | Validation Recall: {:.2f}'.format(
                        self.epoch_mean_recalls[-1],
                        self.validation_recalls[-1]
                    )
                )

                message = '\n'.join(message)

                print(message)

                self.plot(self.losses, save=True)

    def plot(self, data, title=None, xlabel='iter', ylabel='loss', figsize=(16, 14), save=False):

        title = title if title is not None else '{} with {} negatives'.format(self.model_name,
                                                                              self.generate_negatives_type)

        plt.figure(figsize=figsize)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()

        if save:
            plt.savefig('images/{}'.format(title))

    def submission(self, path='submission.csv'):

        pass
