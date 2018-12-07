import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from tools.cleaner import Cleaner
from tools.vocabulary import Vocabulary


class DatasetQuora:

    def __init__(self,
                 train_file,
                 test_file,
                 batch_size=32,
                 sentence_max_length=32,
                 indexing=False,
                 padding_after=True,
                 validation_size=0.2,
                 stratify=False,
                 shuffle=True):

        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.indexing = indexing
        self.validation_size = validation_size
        self.stratify = stratify
        self.shuffle = shuffle

        self.cleaner = Cleaner()

        self.vocabulary = None

        if self.indexing:
            self.vocabulary = Vocabulary(sentence_max_length=sentence_max_length,
                                         padding_after=padding_after,
                                         name='quora_question_pairs',
                                         sos_token=None,
                                         eos_token=None)

        self.train_x = None
        self.train_y = None

        self.validation_x = None
        self.validation_y = None

        self.test_x = None

    def __indexing__(self, text, data_type='train'):

        if data_type == 'train':
            try:
                text = self.vocabulary.collect(input_data=text, output=True, tokenize=True, padding_for_output=True)
            except Exception:
                text = np.NaN
        else:
            try:
                text = self.vocabulary.sentence2indexes(input_data=text, tokenize=True, padding=True)
            except Exception:
                text = np.NaN

        return text

    def __prepare_text__(self, text, data_type='train'):

        text = self.cleaner.clean(sentence=text)

        if self.indexing:
            text = self.__indexing__(text=text, data_type=data_type)
        else:
            text = nltk.tokenize.wordpunct_tokenize(text=text)

        if not text:
            text = np.NaN

        return text

    def collect(self):

        train_data = pd.read_csv(self.train_file, index_col='id')

        train_data.dropna(inplace=True)

        self.train_x, self.validation_x, self.train_y, self.validation_y = train_test_split(
            train_data[[col for col in train_data.columns if col != 'is_duplicate']],
            train_data['is_duplicate'],
            stratify=train_data['is_duplicate'] if self.stratify else None,
            test_size=self.validation_size,
            shuffle=self.shuffle)

        self.train_x.reset_index(inplace=True, drop=True)
        self.train_y.reset_index(inplace=True, drop=True)
        self.validation_x.reset_index(inplace=True, drop=True)
        self.validation_y.reset_index(inplace=True, drop=True)

        self.train_x.question1 = self.train_x.question1.map(lambda x: self.__prepare_text__(text=x, data_type='train'))
        self.train_x.question2 = self.train_x.question2.map(lambda x: self.__prepare_text__(text=x, data_type='train'))

        self.validation_x.question1 = self.validation_x.question1.map(lambda x: self.__prepare_text__(
            text=x,
            data_type='validation'))

        self.validation_x.question2 = self.validation_x.question2.map(lambda x: self.__prepare_text__(
            text=x,
            data_type='validation'))

        self.train_x.dropna(inplace=True)
        self.validation_x.dropna(inplace=True)

        self.train_y = self.train_y[self.train_x.index]
        self.validation_y = self.validation_y[self.validation_x.index]

        self.test_x = pd.read_csv(self.test_file, index_col='test_id')
        self.test_x.drop_duplicates(inplace=True)
        self.test_x.question1 = self.test_x.question1.map(lambda x: self.__prepare_text__(text=x, data_type='test'))
        self.test_x.question2 = self.test_x.question2.map(lambda x: self.__prepare_text__(text=x, data_type='test'))
        self.test_x.fillna(value=['what'])

    def batch_generator(self, data_type='train', batch_size=None):

        data_x = self.__dict__['{}_x'.format(data_type)]
        data_y = self.__dict__['{}_y'.format(data_type)] if data_type != 'test' else None

        batch_size = batch_size if batch_size is not None else self.batch_size

        for n_batch in range(round(data_x.shape[0] / batch_size)):

            question_1 = list(data_x.question1[n_batch * batch_size:(n_batch + 1) * batch_size])
            question_2 = list(data_x.question2[n_batch * batch_size:(n_batch + 1) * batch_size])

            if self.indexing:
                question_1 = np.array([np.array(sample) for sample in question_1])
                question_2 = np.array([np.array(sample) for sample in question_2])

            if data_type == 'test':
                yield question_1, question_2

            target = list(data_y[n_batch * batch_size:(n_batch + 1) * batch_size])

            if self.indexing:
                target = np.array(target)

            yield question_1, question_2, target
