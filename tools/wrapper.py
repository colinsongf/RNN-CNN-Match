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
                 cross_entropy_negative_k_ratio=1.0,
                 validation_batch_size_multiplier=10,
                 generate_negatives_type='random',
                 hard_negatives_multiplier=5,
                 max_hard_negatives=10000,
                 hard_k_next=False):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.available_generate_negatives_types = ['random', 'hard']

        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer

        self.model_name = model_name

        self.batch_size = batch_size

        self.cross_entropy_negative_k_ratio = cross_entropy_negative_k_ratio
        self.validation_batch_size_multiplier = validation_batch_size_multiplier

        self.generate_negatives_type = generate_negatives_type

        self.hard_negatives_multiplier = hard_negatives_multiplier
        self.max_hard_negatives = max_hard_negatives
        self.hard_k_next = hard_k_next

        if self.generate_negatives_type not in self.available_generate_negatives_types:
            raise ValueError('Unknown generate_negatives_type. Available: {}'.format(', '.join(
                self.available_generate_negatives_types)))

        self.cross_entropy_negative_k = int(self.batch_size * self.cross_entropy_negative_k_ratio)

        if self.cross_entropy_negative_k == self.batch_size:
            self.batch_size += self.cross_entropy_negative_k

        self.losses = []
        self.recalls = []
        self.epoch_mean_losses = []
        self.epoch_mean_recalls = []
        self.epochs_passed = 0

        self.validation_losses = []
        self.validation_recalls = []

    # def convert_batch(self, *batch):
    #
    #     batch = [torch.LongTensor(part).to(self.device) for part in batch]
    #
    #     return batch

    def get_random_negatives(self, samples=None):

        samples = samples if samples is not None else self.batch_size

        random_qids = random.sample(self.dataset.qid2question.keys(), samples)

        # return self.dataset.qids2questions(batch=random_qids)
        return self.dataset.prepare_batch(batch=random_qids)

    def get_hard_negatives(self, queries, samples):

        samples = int(min(samples, self.max_hard_negatives))

        # negatives = self.get_random_negatives(samples=samples)

        negatives_qids = random.sample(self.dataset.qid2question.keys(), samples)

        # negatives = self.dataset.qids2questions(batch=negatives_qids)
        negatives = self.dataset.prepare_batch(negatives_qids)

        # query_vactorized = self.model.text_embedding(x=queries)
        query_vactorized = self.model.text_embedding(x=torch.LongTensor(queries).to(self.device))

        # negatives_vectorized = self.model.text_embedding(x=negatives, model_type='candidate')

        negatives_vectorized = self.model.text_embedding(x=torch.LongTensor(negatives).to(self.device),
                                                         model_type='candidate')

        attention = torch.matmul(query_vactorized, negatives_vectorized.transpose(0, 1))

        if self.hard_k_next:

            # if we wont get top-2
            # if we take too many samples we can choose select sample existing in quieries

            max_attentive_matrix = torch.zeros_like(attention)
            values, args = attention.max(dim=1)

            for n, i in enumerate(range(max_attentive_matrix.size(0))):
                max_attentive_matrix[i, args[n]] = values[n]

            attention -= max_attentive_matrix

        max_attentive_indexes = list(attention.argmax(dim=1).cpu().numpy())

        max_attentive_qids = [negatives_qids[n] for n in max_attentive_indexes]

        return self.dataset.prepare_batch(max_attentive_qids)

    def __generate_negatives__(self, queries, batch_size):

        if self.generate_negatives_type == 'random':
            return self.get_random_negatives(samples=batch_size)
        else:
            return self.get_hard_negatives(queries=queries, samples=batch_size * self.hard_negatives_multiplier)

    def __cross_entropy_batch_generator__(self, data, batch_size):

        positives_batch_size = batch_size - self.cross_entropy_negative_k

        for n_batch in range(round(len(data) / batch_size)):

            queries = self.dataset.prepare_batch(
                data[n_batch*positives_batch_size:(n_batch+1)*positives_batch_size].qid1)

            positive_candidates = self.dataset.prepare_batch(
                data[n_batch*positives_batch_size:(n_batch+1)*positives_batch_size].qid2)

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

            yield torch.LongTensor(queries).to(self.device), \
                  torch.LongTensor(candidates).to(self.device),\
                  torch.Tensor(targets).to(self.device)
            # yield self.convert_batch(queries, candidates, targets)
            # yield queries, candidates, targets

    def __triplet_batch_generator__(self, data, batch_size):

        for n_batch in range(round(len(data) / batch_size)):

            queries = self.dataset.prepare_batch(data[n_batch*batch_size:(n_batch+1)*batch_size].qid1)

            positive_candidates = self.dataset.prepare_batch(data[n_batch*batch_size:(n_batch+1)*batch_size].qid2)

            negative_candidates = self.__generate_negatives__(queries=queries,
                                                              batch_size=batch_size)

            indexes = list(range(len(queries)))
            random.shuffle(indexes)

            queries = [queries[index] for index in indexes]
            positive_candidates = [positive_candidates[index] for index in indexes]
            negative_candidates = [negative_candidates[index] for index in indexes]

            yield torch.LongTensor(queries).to(self.device), \
                  torch.LongTensor(positive_candidates).to(self.device), \
                  torch.LongTensor(negative_candidates).to(self.device)
            # yield self.convert_batch(queries, positive_candidates, negative_candidates)
            # yield queries, positive_candidates, negative_candidates

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

                pbar = tqdm(total=total_n_batches, desc='Train Epoch {}'.format(self.epochs_passed + 1))

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

                # maybe is not necessary
                if self.model.loss_type == 'cross_entropy':
                    # TODO max as hyperparameter
                    torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), 1.0)

                self.optimizer.step()

                if verbose:
                    pbar.update(1)

            self.epochs_passed += 1

            self.epoch_mean_losses.append(np.mean(batch_losses))
            self.epoch_mean_recalls.append(np.mean(batch_recalls))

            validation_epoch_mean_loss = []
            validation_epoch_mean_recall = []

            validation_batch_size = self.batch_size * self.validation_batch_size_multiplier

            for batch in self.batch_generator(data_type='validation', batch_size=validation_batch_size):

                with torch.no_grad():

                    # validation_loss, vectorized_batch = self.model.compute_loss(*batch)
                    # validation_recall = self.model.compute_recall(*vectorized_batch)

                    validation_loss, validation_recall = self.compute_loss_recall(batch)

                    validation_epoch_mean_loss.append(validation_loss.item())
                    validation_epoch_mean_recall.append(validation_recall)

            self.validation_losses.append(np.mean(validation_epoch_mean_loss))
            self.validation_recalls.append(np.mean(validation_epoch_mean_recall))

            if verbose:
                pbar.close()

                message = list()

                message.append(
                    'Epoch: [{}/{}] | {} loss: {:.3f} | Validation Loss: {:.3f}'.format(
                        n_epoch,
                        epochs,
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

        if verbose:
            self.plot(self.losses, save=True)

    def plot(self, data, title=None, xlabel='iter', ylabel='loss', figsize=(16, 14), save=True):

        title = title if title is not None else '{} {} with {} negatives'.format(self.model_name,
                                                                                 self.model.loss_type,
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
