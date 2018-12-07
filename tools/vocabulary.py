import numpy as np
import collections
import nltk
import json
from tqdm import tqdm as tqdm


class Vocabulary:

    def __init__(self,
                 sentence_max_length=None,
                 ignore_sentence_max_length_for_collect=True,
                 padding_after=True,
                 collect_pos_tags_flag=False,
                 stop_words=None,
                 name='default_vocabulary',
                 language=None,
                 unk_token='UNK',
                 pad_token='PAD',
                 sos_token='SOS',
                 eos_token='EOS'):

        self.sentence_max_length = sentence_max_length
        self.ignore_sentence_max_length_for_collect = ignore_sentence_max_length_for_collect
        self.padding_after = padding_after
        self.collect_pos_tags_flag = collect_pos_tags_flag
        self.stop_words = stop_words if stop_words is not None else []

        self.name = name
        self.language = language

        self.addition_length = 0

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        if sos_token is not None:
            self.addition_length += 1

        if eos_token is not None:
            self.addition_length += 1

        if self.sentence_max_length is not None:
            self.sentence_max_length -= self.addition_length

        self.unique_tokens = [self.__dict__[key] for key in self.__dict__
                              if key[-5:] == 'token' and self.__dict__[key] is not None]

        self.token2index = {}
        self.index2token = {}

        self.global_token_count = 0
        self.global_document_count = 0
        self.chars = {}
        self.global_document_lengths = []

        self.init_dictionaries()

        self.deleted_tokens = []
        self.deleted_indexes = []

    def init_dictionaries(self, **kwargs):

        for key in kwargs:
            if key in self.__dict__:
                self.__dict__[key] = kwargs[key]

        self.token2index = {
            key: self.__init_token2index_sample__(token=key, index=index, token_count=0, document_count=0)
            for index, key in enumerate(self.unique_tokens)
        }

        self.index2token = {index: key for index, key in enumerate(self.unique_tokens)}

        self.global_token_count = 0
        self.global_document_count = 0
        self.chars = {}

        for token in self.unique_tokens:
            self.__collect_chars__(input_data=token)

    @property
    def get_tokens2index(self):

        return {token: self.token2index[token]['index'] for token in self.token2index}

    def __collect_chars__(self, input_data, n_token=1):

        for char in input_data:
            if char in self.chars:
                self.chars[char] += n_token
            else:
                self.chars[char] = n_token

    def __get_pos_tag__(self, token):

        # TODO need better pos tagger

        return nltk.pos_tag(tokens=[token], lang=self.language)[0][1]

    def __init_token2index_sample__(self, token, index,
                                    token_count=1, document_count=1,
                                    token_frequency=0., document_frequency=0.):

        output_dict = {
            'index': index,
            'token_count': token_count,
            'document_count': document_count,
            'token_frequency': token_frequency,
            'document_frequency': document_frequency
        }

        if self.collect_pos_tags_flag:

            if self.language is None:
                raise ValueError('Set language attribute in self.__dict__')

            output_dict['pos_tag'] = self.__get_pos_tag__(token)

        if self.stop_words:
            output_dict['is_stop_word'] = True if token in self.stop_words else False

        return output_dict

    def __len__(self):

        return len(self.token2index)

    def __getitem__(self, item):
        # TODO add numpy types
        if type(item) == int:

            try:
                return self.index2token[item]
            except KeyError:
                return self.unk_token

        else:

            if item in self.token2index:
                return self.token2index[item]['index']
            else:
                return self.token2index[self.unk_token]['index']

    def crop(self, input_data, sentence_max_length=None):

        sentence_max_length = sentence_max_length if sentence_max_length is not None else self.sentence_max_length

        eos_marker = self.eos_token if type(input_data[0]) == str else self.token2index[self.eos_token]['index']
        pad_marker = self.pad_token if type(input_data[0]) == str else self.token2index[self.pad_token]['index']

        if len(input_data) >= sentence_max_length:

            if eos_marker in input_data:
                input_data = [token for token in input_data if token != pad_marker and token != eos_marker]
                input_data = input_data[:sentence_max_length - 1] + [eos_marker]
                if len(input_data) < sentence_max_length:
                    input_data += [pad_marker] * (sentence_max_length - len(input_data))
            else:
                input_data = input_data[:sentence_max_length]

        return input_data

    def add_sos_eos_tokens(self, input_data):

        if self.sos_token is not None:

            sos_marker = self.sos_token if type(input_data[0]) == str else self.token2index[self.sos_token]['index']

            if sos_marker not in input_data:
                input_data.insert(0, self.sos_token)

        if self.eos_token is not None:

            eos_marker = self.eos_token if type(input_data[0]) == str else self.token2index[self.eos_token]['index']

            if eos_marker not in input_data:
                input_data.append(self.eos_token)

        return input_data

    def padding(self, input_data, sentence_max_length=None):

        sentence_max_length = sentence_max_length if sentence_max_length is not None else self.sentence_max_length

        if sentence_max_length:

            input_data = self.crop(input_data=input_data, sentence_max_length=sentence_max_length)

            pad_marker = self.pad_token if type(input_data[0]) == str else self.token2index[self.pad_token]['index']

            if self.padding_after:
                input_data += [pad_marker] * (sentence_max_length - len(input_data))
            else:
                input_data = ([pad_marker] * (sentence_max_length - len(input_data))) + input_data

        return input_data

    def collect(self,
                input_data,
                tokenize=False,
                output=False,
                indexes=True,
                add_eos_sos_tokens=False,
                padding_for_output=True):

        input_data = self.sentence2tokens(input_data=input_data,
                                          tokenize=tokenize,
                                          add_eos_sos_tokens=add_eos_sos_tokens,
                                          padding=False)

        self.global_document_lengths.append(len(input_data))

        counter = collections.Counter()

        for token in input_data:
            counter[token] += 1

        for token in counter:

            self.__collect_chars__(input_data=token, n_token=counter[token])

            if token in self.token2index:

                self.token2index[token]['token_count'] += counter[token]
                self.token2index[token]['document_count'] += 1

            else:

                current_index_token = len(self.token2index)

                self.token2index[token] = self.__init_token2index_sample__(token=token,
                                                                           index=current_index_token,
                                                                           token_count=counter[token])

                self.index2token[current_index_token] = token

            self.global_token_count += counter[token]

        self.global_document_count += 1

        if output:

            if padding_for_output:
                input_data = self.padding(input_data=input_data)

            if indexes:
                input_data = self.tokens2indexes(input_data=input_data)

            return input_data

    def delete_token(self, input_data):

        delete_chars = []

        tmp_token_count = self.token2index[input_data]['token_count']

        for char in input_data:

            self.chars[char] -= tmp_token_count

            if self.chars[char] <= 0:
                delete_chars.append(char)

        self.global_token_count -= tmp_token_count

        for char in delete_chars:
            del self.chars[char]

        self.deleted_tokens.append(input_data)
        self.deleted_indexes.append(self.token2index[input_data]['index'])
        del self.index2token[self.token2index[input_data]['index']]
        del self.token2index[input_data]

    def sentence2tokens(self, input_data, tokenize=False, add_eos_sos_tokens=False, padding=False):

        if tokenize:
            input_data = nltk.tokenize.wordpunct_tokenize(input_data)

        if add_eos_sos_tokens:
            input_data = self.add_sos_eos_tokens(input_data=input_data)

        if self.sentence_max_length:

            if padding:
                input_data = self.padding(input_data=input_data)
            elif not self.ignore_sentence_max_length_for_collect:
                input_data = self.crop(input_data=input_data)

        return input_data

    def tokens2indexes(self, input_data):

        return [self.__getitem__(item=token) for token in input_data]

    def sentence2indexes(self, input_data, tokenize=False, add_eos_sos_tokens=False, padding=False):

        input_data = self.sentence2tokens(input_data=input_data,
                                          tokenize=tokenize,
                                          add_eos_sos_tokens=add_eos_sos_tokens,
                                          padding=padding)

        input_data = self.tokens2indexes(input_data=input_data)

        return input_data

    def indexes2tokens(self, input_data):

        return self.tokens2indexes(input_data=input_data)

    def count(self, item):

        if type(item) == str:
            return self.token2index[item]['token_count']
        elif type(item) == int:
            return self.token2index[self.index2token[item]]['token_count']

    def frequency(self, item):

        if type(item) == str:
            return self.token2index[item]['token_frequency']
        elif type(item) == int:
            return self.token2index[self.index2token[item]]['token_frequency']

    def calculate_frequency(self):

        if self.global_token_count == 0 or self.global_document_count == 0:
            raise ZeroDivisionError(
                'Need collect corpus to calculate frequency. '
                'Now token count is {} and document count is {}.'.format(self.global_token_count,
                                                                         self.global_document_count))

        for token in self.token2index:

            self.token2index[token]['token_frequency'] = self.token2index[token]['token_count'] \
                                                         / self.global_token_count

            self.token2index[token]['document_frequency'] = self.token2index[token]['document_count'] \
                                                            / self.global_document_count

    def filter(self,
               min_token_count=0,
               max_token_count=None,
               min_document_count=0,
               max_document_count=None,
               min_token_frequency=0.,
               max_token_frequency=None,
               min_document_frequency=0.,
               max_document_frequency=None,
               calculate_frequency=True,
               verbose=False):

        compare_data = {
            'token_count': {
                'min': min_token_count,
                'max': max_token_count if max_token_count is not None else self.global_token_count
            },
            'document_count': {
                'min': min_document_count,
                'max': max_document_count if max_document_count is not None else self.global_document_count
            },
            'token_frequency': {
                'min': min_token_frequency,
                'max': max_token_frequency if max_token_frequency is not None else 1.
            },
            'document_frequency': {
                'min': min_document_frequency,
                'max': max_document_frequency if max_document_frequency is not None else 1.
            }
        }

        if calculate_frequency:
            self.calculate_frequency()

        tokens = tqdm(list(self.token2index.keys()), desc='Filter tokens') if verbose else list(self.token2index.keys())

        for token in tokens:

            delete_status = False

            for field in compare_data:
                condition = compare_data[field]['min'] <= self.token2index[token][field] <= compare_data[field]['max']
                delete_status = delete_status or not condition

            if delete_status and token not in self.unique_tokens:

                self.delete_token(input_data=token)

    def save(self, filename, beautify=False):

        data = {key: self.__dict__[key] for key in self.__dict__}

        data = json.dumps(data, ensure_ascii=False, indent=4) if beautify else json.dumps(data, ensure_ascii=False)

        with open(file=filename, mode='w', encoding='utf-8') as f:
            f.write(data)

    def load(self, filename):

        with open(file=filename, mode='r', encoding='utf-8') as f:
            self.__dict__ = json.loads(f.read())

        self.index2token = {int(index): token for index, token in self.index2token.items()}

    def summary(self):

        output = list()

        output.append('Vocabulary name: {}'.format(self.name))
        if self.language is not None:
            output.append('Language: {}'.format(self.language))
        output.append('')
        output.append('Unique tokens: {}'.format(len(self.token2index)))
        output.append('Token count: {}'.format(self.global_token_count))
        output.append('Document count: {}'.format(self.global_document_count))
        output.append('')
        output.append('Unique chars: {}'.format(len(self.chars)))
        output.append('')
        output.append('Max document length: {}'.format(max(self.global_document_lengths)))
        output.append('Mean document length: {:.1f}'.format(np.mean(self.global_document_lengths)))
        output.append('Median document length: {:.1f}'.format(np.median(self.global_document_lengths)))
        output.append('Min document length: {}'.format(min(self.global_document_lengths)))

        print('\n'.join(output))
