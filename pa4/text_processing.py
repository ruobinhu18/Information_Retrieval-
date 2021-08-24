import re
from typing import Any, List
from math import log2
from nltk.tokenize import word_tokenize  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore
from nltk.corpus import stopwords  # type: ignore


class TextProcessing:
    def __init__(self, stemmer, stop_words, *args):
        """
        class TextProcessing is used to tokenize and normalize tokens that will be further used to build inverted index.
        :param stemmer:
        :param stop_words:
        :param args:
        """
        self.stemmer = stemmer
        self.STOP_WORDS = stop_words

    @classmethod
    def from_nltk(
            cls,
            stemmer: Any = PorterStemmer().stem,
            stop_words: List[str] = stopwords.words("english"),
    ) -> "TextProcessing":
        """
        initialize from nltk
        :param stemmer:
        :param stop_words:
        :return:
        """
        return cls(stemmer, set(stop_words))

    def normalize(self, token: str) -> str:
        """
        normalize the token based on:
        1. make all characters in the token to lower case
        2. remove any characters from the token other than alphanumeric characters and dash ("-")
        3. after step 1, if the processed token appears in the stop words list or its length is 1, return an empty string
        4. after step 1, if the processed token is NOT in the stop words list and its length is greater than 1, return the stem of the token
        :param token:
        :return:
        """
        # first process
        token = token.lower()
        # second process
        token = re.sub(r'[^a-z0-9\-]+', '', token)
        # third process
        if (token in self.STOP_WORDS) or (len(token) == 1):
            return ""
        else:
            return self.stemmer(token)

    def get_normalized_tokens(self, title: str, content: str) -> List[str]:
        """
        pass in the title and content_str of each document, and return a list of normalized tokens (exclude the empty string)
        you may want to apply word_tokenize first to get un-normalized tokens first.
        Note that you don't want to remove duplicate tokens as what you did in HW3, because it will later be used to compute term frequency
        :param title:
        :param content:
        :return:
        """
        if title is not None:
            title = word_tokenize(title)
            title = [self.normalize(token) for token in title]
        else:
            title = []
        if content is not None:
            content = word_tokenize(content)
            content = [self.normalize(token) for token in content]
        else:
            content = []
        s = title + content
        # remove empty string
        s = [e for e in s if e != '']
        return sorted(s)

    @staticmethod
    def idf(N: int, df: int) -> float:
        """
        compute the logarithmic (base 2) idf score
        :param N: document count N
        :param df: document frequency
        :return: log2(N/df)
        """
        return log2(N / df)

    @staticmethod
    def tf(freq: int) -> float:
        """
        compute the logarithmic tf (base 2) score
        :param freq: raw term frequency
        :return: log tf score
        """
        return 1 + log2(freq)


if __name__ == "__main__":
    pass
