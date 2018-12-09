import numpy as np
import torch
from tqdm import tqdm
import math
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
                 max_norm=None,
                 batch_size=32,
                 cross_entropy_negative_k_ratio=1.0,
                 validation_batch_size_multiplier=10,
                 generate_negatives_type='random',
                 hard_negatives_multiplier=5,
                 max_hard_negatives=10000,
                 hard_k_next=True):
        """
        Init params
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.available_generate_negatives_types = ['random', 'hard']

        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer

        self.model_name = model_name

        self.max_norm = max_norm

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

        if self.model.loss_type == 'cross_entropy':
            self.batch_size += self.cross_entropy_negative_k

        self.model_name = '{}_{}_with_{}_negatives'.format(self.model_name,
                                                           self.model.loss_type,
                                                           self.generate_negatives_type)

        self.losses = []
        self.recalls = []
        self.epoch_mean_losses = []
        self.epoch_mean_recalls = []
        self.epochs_passed = 0

        self.validation_losses = []
        self.validation_recalls = []

        self.best_mean_loss = 1000
        self.validation_best_mean_loss = 1000

    def get_random_negatives(self, samples=None):
        """
        Generate random negatives candidates
        """

        samples = samples if samples is not None else self.batch_size

        random_qids = random.sample(self.dataset.qid2question.keys(), samples)

        return self.dataset.prepare_batch(batch=random_qids)

    def get_hard_negatives(self, queries, samples):
        """
        Generate hard negatives candidates
        """

        samples = int(min(samples, self.max_hard_negatives))

        negatives_qids = random.sample(self.dataset.qid2question.keys(), samples)

        negatives = self.dataset.prepare_batch(negatives_qids)

        query_vactorized = self.model.text_embedding(x=torch.LongTensor(queries).to(self.device))

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
        """
        Choose method and generate negatives candidates
        """

        if self.generate_negatives_type == 'random':
            return self.get_random_negatives(samples=batch_size)
        else:
            return self.get_hard_negatives(queries=queries, samples=batch_size * self.hard_negatives_multiplier)

    def __cross_entropy_batch_generator__(self, data, batch_size):
        """
        Batch generator for binary cross-entropy loss
        """

        positives_batch_size = batch_size - self.cross_entropy_negative_k

        for n_batch in range(round(len(data) / batch_size)):

            queries = self.dataset.prepare_batch(
                data[n_batch*positives_batch_size:(n_batch+1)*positives_batch_size].qid1)

            positive_candidates = self.dataset.prepare_batch(
                data[n_batch*positives_batch_size:(n_batch+1)*positives_batch_size].qid2)

            negative_candidates = self.__generate_negatives__(queries=queries,
                                                              batch_size=self.cross_entropy_negative_k)

            targets = [1 for _ in range(len(positive_candidates))] + [0 for _ in range(len(negative_candidates))]

            queries *= (math.ceil(2 + self.cross_entropy_negative_k_ratio) + 1)
            queries = queries[:len(targets)]

            candidates = positive_candidates + negative_candidates

            indexes = list(range(len(targets)))
            random.shuffle(indexes)

            queries = [queries[index] for index in indexes]
            candidates = [candidates[index] for index in indexes]
            targets = [targets[index] for index in indexes]

            yield torch.LongTensor(queries).to(self.device), \
                  torch.LongTensor(candidates).to(self.device),\
                  torch.Tensor(targets).to(self.device)

    def __triplet_batch_generator__(self, data, batch_size):
        """
        Batch generator for Triplet Margin Loss loss
        """

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

    def batch_generator(self, data_type='train', batch_size=None):
        """
        Batch generator with choosing method and select data
        """

        data = self.dataset.__dict__[data_type]
        batch_size = batch_size if batch_size is not None else self.batch_size

        if self.model.loss_type == 'cross_entropy':
            return self.__cross_entropy_batch_generator__(data=data, batch_size=batch_size)
        elif self.model.loss_type == 'triplet':
            return self.__triplet_batch_generator__(data=data, batch_size=batch_size)

    def compute_loss_recall(self, batch, validation=False):
        """
        Compute loss and recall
        """

        loss, vectorized_batch = self.model.compute_loss(*batch)
        recall = self.model.compute_recall(*vectorized_batch, mean=not validation)

        return loss, recall

    def train(self,
              epochs=5,
              negatives_type=None,
              verbose=False,
              save_best=False):
        """
        Train method with loss and metric tracking and ploting results
        """

        self.generate_negatives_type = negatives_type if negatives_type is not None else self.generate_negatives_type

        for n_epoch in range(1, epochs+1):

            if verbose:

                if self.model.loss_type == 'cross_entropy':
                    total_n_batches = len(self.dataset.train) // (self.batch_size - self.cross_entropy_negative_k)
                else:
                    total_n_batches = len(self.dataset.train) // self.batch_size

                pbar = tqdm(total=total_n_batches, desc='Train Epoch {}'.format(self.epochs_passed + 1))

            batch_losses = []
            batch_recalls = []

            for batch in self.batch_generator(data_type='train'):

                loss, recall = self.compute_loss_recall(batch)

                batch_recalls.append(recall)
                self.recalls.append(recall)

                self.losses.append(loss.item())
                batch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()

                if self.max_norm is not None:
                    torch.nn.utils.clip_grad.clip_grad_norm(self.model.parameters(), max_norm=self.max_norm)

                self.optimizer.step()

                if verbose:
                    pbar.update(1)

            self.epochs_passed += 1

            self.epoch_mean_losses.append(np.mean(batch_losses))
            self.epoch_mean_recalls.append(np.mean(batch_recalls))

            validation_epoch_mean_loss = []
            validation_epoch_mean_recalls = []

            validation_batch_size = self.batch_size * self.validation_batch_size_multiplier

            for batch in self.batch_generator(data_type='validation', batch_size=validation_batch_size):

                with torch.no_grad():

                    validation_loss, validation_recall = self.compute_loss_recall(batch, validation=True)

                    validation_epoch_mean_loss.append(validation_loss.item())
                    validation_epoch_mean_recalls.extend(validation_recall)

            self.validation_losses.append(np.mean(validation_epoch_mean_loss))
            self.validation_recalls.append(np.mean(validation_epoch_mean_recalls))

            # if save_best and self.epoch_mean_losses[-1] < self.best_mean_loss:
            #     self.best_mean_loss = self.epoch_mean_losses[-1]
            #     self.save_model()

            if save_best and self.validation_losses[-1] < self.best_mean_loss:
                self.best_mean_loss = self.validation_losses[-1]
                self.save_model()

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

    def save_model(self, path=None):
        """
        Save model. Use in train for save best perfomance model
        """

        path = path if path is not None else self.model_name

        # TODO add more info about model
        torch.save(self.model, path)

    def plot(self, data, title=None, xlabel='iter', ylabel='loss', figsize=(16, 14), save=True):
        """
        Plotting data
        """

        title = title if title is not None else self.model_name

        plt.figure(figsize=figsize)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()

        if save:
            plt.savefig('images/{}'.format(title))

    def submission(self, path=None, batch_size=2048, verbose=False):
        """
        Generate submission file
        """

        if path is None:
            path = '{}_submission.csv'.format(self.model_name)

        test, submission = self.dataset.get_test_submission

        is_duplicate = []

        total_n_batches = round(test.shape[0] / batch_size) + 1

        if verbose:

            pbar = tqdm(total=total_n_batches, desc='Train Epoch {}'.format(self.epochs_passed + 1))

        for n_batch in range(total_n_batches):

            que1 = test.question1[n_batch * batch_size:(n_batch + 1) * batch_size]
            que2 = test.question2[n_batch * batch_size:(n_batch + 1) * batch_size]

            que1 = self.dataset.prepare_batch(list(que1), qids=False)
            que2 = self.dataset.prepare_batch(list(que2), qids=False)

            que1 = torch.LongTensor(que1).to(self.device)
            que2 = torch.LongTensor(que2).to(self.device)

            que1 = self.model.text_embedding(que1)
            que2 = self.model.text_embedding(que2)

            similarity = self.model.similarity_function(que1, que2)

            similarity = torch.nn.functional.relu(similarity - self.model.eps).cpu().numpy()

            is_duplicate.extend([float(sim) for sim in similarity])

            if verbose:
                pbar.update(1)

        submission.is_duplicate = is_duplicate

        submission.to_csv(path, index=None)
