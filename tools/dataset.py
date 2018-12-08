import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from tools.cleaner import Cleaner
from tools.vocabulary import Vocabulary


class DatasetQuora:

    def __init__(self,
                 train_file,
                 test_file,
                 sample_submission_file,
                 sentence_max_length=32,
                 word_indexing=False,
                 padding_after=True,
                 validation_size=0.2,
                 shuffle=True,
                 text_fillna='what?'):

        self.train_file = train_file
        self.test_file = test_file
        self.sample_submission_file = sample_submission_file
        self.word_indexing = word_indexing
        self.validation_size = validation_size
        self.shuffle = shuffle
        self.text_fillna = text_fillna

        self.qid2question = {}

        self.cleaner = Cleaner()

        if self.word_indexing:
            self.vocabulary = Vocabulary(sentence_max_length=sentence_max_length,
                                         padding_after=padding_after,
                                         name='quora_question_pairs',
                                         sos_token=None,
                                         eos_token=None)
        else:
            self.vocabulary = None

        self.train = None
        self.validation = None

    def __word_indexing__(self, text, data_type='train'):

        if data_type == 'train':
            try:
                text = self.vocabulary.collect(input_data=text, output=True, tokenize=True, padding_for_output=True)
            except Exception:
                text = [self.vocabulary.pad_token]
        else:
            try:
                text = self.vocabulary.sentence2indexes(input_data=text, tokenize=True, padding=True)
            except Exception:
                text = [self.vocabulary.pad_token]

        return text

    def __prepare_text__(self, text, data_type='train'):

        text = self.cleaner.clean(x=text)

        if self.word_indexing:
            text = self.__word_indexing__(text=text, data_type=data_type)
        else:
            text = nltk.tokenize.wordpunct_tokenize(text=text)

        if not text:
            text = [self.text_fillna]

        return text

    @property
    def existing_words(self):

        return list(self.vocabulary.token2index.keys())

    def qids2questions(self, batch):

        return [self.qid2question[sample] for sample in batch]

    def collect(self):

        train_data = pd.read_csv(self.train_file, index_col='id')

        # qids = list(train_data.qid1) + list(train_data.qid2)
        # qids = list(set(qids))
        # qids.sort()

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

    @property
    def get_test_submission(self):

        test = pd.read_csv(self.test_file)
        sample_submission = pd.read_csv(self.sample_submission_file)
        test.drop_duplicates(inplace=True)
        test = test[test.test_id.isin(sample_submission.test_id)]

        test.question1 = test.question1.map(lambda x: self.__prepare_text__(text=x, data_type='test'))
        test.question2 = test.question2.map(lambda x: self.__prepare_text__(text=x, data_type='test'))

        return test, sample_submission
