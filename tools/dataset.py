import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from tools.cleaner import Cleaner
import numpy as np
from tqdm import tqdm
import torch


class DatasetQuora:

    def __init__(self,
                 train_file,
                 test_file,
                 sample_submission_file,
                 sequence_max_length=32,
                 padding_after=True,
                 pad_token='PAD',
                 pad_index=0,
                 validation_size=0.2,
                 shuffle=True,
                 text_fillna='what?'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_file = train_file
        self.test_file = test_file
        self.sample_submission_file = sample_submission_file

        self.sequence_max_length = sequence_max_length

        self.padding_after = padding_after
        self.pad_token = pad_token
        self.pad_index = pad_index

        self.validation_size = validation_size
        self.shuffle = shuffle
        self.text_fillna = text_fillna

        self.cleaner = Cleaner()

        self.qid2question = {}
        self.token2index = {}
        self.index2token = {}

        self.train = None
        self.validation = None

        self.collect()

    def __prepare_text__(self, text, data_type='train'):

        text = self.cleaner.clean(x=text)

        if not text:
            text = [self.text_fillna]

        text = nltk.tokenize.wordpunct_tokenize(text=text)

        if data_type == 'train':

            for token in text:

                index = len(self.token2index)

                if token not in self.token2index:
                    self.token2index[token] = index
                    self.index2token[index] = token

        return text

    def qids2questions(self, batch):

        return [self.qid2question[sample] for sample in batch]

    def prepare_batch(self, batch, qids=True):

        if qids:
            batch = self.qids2questions(batch=batch)

        for n_sample in range(len(batch)):

            tokens = batch[n_sample]

            tokens = [self.token2index[token] for token in tokens if token in self.token2index]

            tokens = tokens[:self.sequence_max_length]

            if len(tokens) < self.sequence_max_length:

                pads = [self.pad_index] * (self.sequence_max_length - len(tokens))

                if self.padding_after:
                    tokens = tokens + pads
                else:
                    tokens = pads + tokens

            batch[n_sample] = tokens

        return batch

    def collect(self):

        train_data = pd.read_csv(self.train_file, index_col='id')

        train_data.question1 = train_data.question1.map(lambda x: self.__prepare_text__(text=x, data_type='train'))
        train_data.question2 = train_data.question2.map(lambda x: self.__prepare_text__(text=x, data_type='train'))

        train_data.dropna(inplace=True)

        for index in train_data.index:

            qid1 = int(train_data.qid1[index])
            qid2 = int(train_data.qid2[index])

            if qid1 not in self.qid2question:
                self.qid2question[qid1] = train_data.question1[index]

            if qid2 not in self.qid2question:
                self.qid2question[qid2] = train_data.question2[index]

        train_data = train_data[train_data.is_duplicate == 1][['qid1', 'qid2']]

        self.train, self.validation = train_test_split(train_data,
                                                       test_size=self.validation_size,
                                                       shuffle=self.shuffle)

    def load_pretrained_embeddings(self, embedding_weight_file, embedding_size=300, verbose=False):

        self.token2index = {
            self.pad_token: self.pad_index
        }

        self.index2token = {
            self.pad_index: self.pad_token
        }

        embedding_matrix = [np.zeros(shape=(embedding_size,))]

        with open(file=embedding_weight_file, mode='r', encoding='utf-8', errors='ignore') as file:

            index = len(self.token2index)

            lines = tqdm(file.readlines(), desc='Collect embeddings') if verbose else file.readlines()

            for line in lines:

                line = line.split()

                token = ' '.join(line[:-embedding_size])
                embeddings = np.asarray(line[-embedding_size:], dtype='float32')

                if not token or embeddings.shape[0] != embedding_size:
                    continue

                self.token2index[token] = index
                self.index2token[index] = token

                embedding_matrix.append(embeddings)

                index += 1

        return embedding_matrix

        # self.embedding_layer = torch.nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix)).to(self.device)

    @property
    def get_test_submission(self):

        test = pd.read_csv(self.test_file)
        sample_submission = pd.read_csv(self.sample_submission_file)
        test.drop_duplicates(inplace=True)
        test = test[test.test_id.isin(sample_submission.test_id)]

        test.question1 = test.question1.map(lambda x: self.__prepare_text__(text=x, data_type='test'))
        test.question2 = test.question2.map(lambda x: self.__prepare_text__(text=x, data_type='test'))

        test.reset_index(inplace=True, drop=True)

        sample_submission.is_duplicate = sample_submission.is_duplicate.astype('float')

        return test, sample_submission
