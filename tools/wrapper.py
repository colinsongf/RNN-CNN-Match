import numpy as np
import random
import torch
from tqdm import tqdm
import tools


class Wrapper(tools.DatasetQuora):

    def __init__(self,
                 train_file,
                 test_file,
                 model,
                 optimizer,
                 model_name='default_model',
                 batch_size=32,
                 validation_size=0.2,
                 sentence_max_length=32,
                 indexing=False,
                 padding_after=True,
                 stratify=False,
                 shuffle=True):

        super(Wrapper, self).__init__(train_file=train_file,
                                      test_file=test_file,
                                      batch_size=batch_size,
                                      sentence_max_length=sentence_max_length,
                                      indexing=indexing,
                                      padding_after=padding_after,
                                      validation_size=validation_size,
                                      stratify=stratify,
                                      shuffle=shuffle)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.model_name = model_name
        self.optimizer = optimizer

        self.losses = []
        self.batch_mean_losses = []

        self.epochs_passed = 0

        self.collect()

    # def __get_indexes__(self, data_type='train'):
    #
    #     if data_type == 'train':
    #         indexes = self.indexes[:self.train_test_separator]
    #     elif data_type == 'test':
    #         indexes = self.indexes[self.train_test_separator:]
    #     else:
    #         raise ValueError('Data types available: train and test')
    #
    #     return indexes

    # def batch_generator(self,
    #                     data_type='train',
    #                     negatives_type='random',
    #                     k_negatives=150,
    #                     sequence_max_length=None):
    #
    #     indexes = self.__get_indexes__(data_type=data_type)
    #
    #     for n_batch in range(len(indexes) // self.batch_size):
    #
    #         batch_indexes = indexes[n_batch * self.batch_size:(n_batch + 1) * self.batch_size]
    #
    #         sequence_max_length = sequence_max_length if sequence_max_length is not None else -1
    #
    #         questions = [self.questions[index][:sequence_max_length] for index in batch_indexes]
    #         answers_positive = [self.answers[index][:sequence_max_length] for index in batch_indexes]
    #
    #         if negatives_type == 'random':
    #             answers_negatives = self.get_random_answers(data_type=data_type)
    #         elif negatives_type == 'with_gradient':
    #             answers_negatives = self.get_with_gradient_answers(data_type=data_type)
    #         elif negatives_type == 'hard':
    #             answers_negatives = self.get_hard_answers(data_type=data_type)
    #         elif negatives_type == 'semi-hard':
    #             answers_negatives = self.get_semi_hard_answers(data_type=data_type)
    #         else:
    #             answers_negatives = []
    #
    #         yield questions, answers_positive, answers_negatives

    def get_random_answers(self, data_type='train'):

        # TODO random answers

        pass

    def get_with_gradient_answers(self, data_type='train'):

        # TODO negatives with gradient

        pass

    def get_hard_answers(self, data_type='train'):

        # TODO hard negatives

        pass

    def get_semi_hard_answers(self, data_type='train'):

        # TODO semi hard negatives

        pass

    def train(self,
              epochs=5,
              negatives_type='random',
              verbose=False):

        self.losses = []
        self.batch_mean_losses = []

        for n_epoch in range(1, epochs+1):

            if verbose:
                pbar = tqdm(total=len(self.train_x) // self.batch_size, desc='Train Epoch {}'.format(n_epoch))

            batch_losses = []

            for query, candidate, target in self.batch_generator(data_type='train'):

                target = torch.Tensor(target).to(self.device)

                query, candidate = self.model(query, candidate)

                loss = self.model.compute_cross_entropy(query=query, candidate=candidate, target=target)

                self.losses.append(loss.item())
                batch_losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if verbose:
                    pbar.update(1)

            batch_mean_loss = np.mean(batch_losses)

            self.batch_mean_losses.append(batch_mean_loss)

            if verbose:
                pbar.close()

            message = 'Epoch: [{}/{}] | Loss: {:.5f}'.format(
                n_epoch,
                epochs,
                batch_mean_loss
            )

            if verbose:
                print(message)

            self.epochs_passed += 1
