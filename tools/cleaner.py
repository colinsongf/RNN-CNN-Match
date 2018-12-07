import re
import unicodedata


class Cleaner:

    def __init__(self):

        pass

    @staticmethod
    def unicode_to_ascii(x):

        return ''.join(
            c for c in unicodedata.normalize('NFD', x)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalize_string(x):

        # x = re.sub(r"([.!?])", r" .", x)
        # x = re.sub('[0-9]{5,}', '##### ', x)
        x = re.sub('[0-9]{4,}', '#### ', x)
        x = re.sub('[0-9]{3}', '### ', x)
        x = re.sub('[0-9]{2}', '## ', x)
        s = re.sub(r"[^а-яА-Яa-zA-Z!?)(#]+", r" ", x)

        return s

    def clean(self, x):

        x = str(x).strip().lower()
        x = self.unicode_to_ascii(x)
        x = self.normalize_string(x)

        return x
