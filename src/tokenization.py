import torch
import spacy
import random
import numpy as np
from spacy.tokens.doc import Doc
from abc import ABC, abstractmethod
from spacy.tokens.token import Token
from datasets import load_dataset, Dataset

SEED = 42


def set_seed(seed: int):
    """
    This function sets the seed for the random number generators in Python's built-in random module, NumPy, 
    PyTorch CPU, and PyTorch GPU. This is useful for ensuring reproducibility of results. 

    Args:
        seed (int): The seed to set the random number generators to.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Tokenization(ABC):

    def __init__(self, words: list):
        """
        Args:
            words(list): A list of target words.
        """
        self.words = words

    @abstractmethod
    def is_doc_relevant(self, row: object, token: object) -> bool:
        """
        Abstract method to determine whether a token in a given row of a dataset is relevant for tokenization.

        Args:
            row (object): An object representing a row of a dataset.
            token (object): An object representing a token in the row.

        Returns:
            bool: True if the token is relevant for tokenization, False otherwise.
        """

    def tokenize(self, dataset: str, sampling: int = 0) -> list:
        """
        Abstract method to tokenize the sentences in the given dataset.

        Args:
            dataset (str): A string representing the path to the dataset to be tokenized.
            sampling (int): An integer representing the number of sentences to randomly sample from the dataset (default is 0).

        Returns: list: A list of dictionaries containing tokenization information, including sentence index, lemma, 
        token text, start and end indices, and the original sentence string. 
        """
        # load dataset
        dataset = load_dataset('text', data_files=dataset, split='train')['text']

        # random sampling
        if sampling:
            dataset = self._random_sampling(dataset, sampling)

        # collect data
        rows = list()
        for idx, row in enumerate(dataset):
            row = row.strip()

            for i, token in enumerate(row.split()):
                if self.is_doc_relevant(row, token):
                    start = len(" ".join(row[:i]))
                    rows.append(dict(sentidx=idx,
                                     lemma=token,
                                     token=token,
                                     start=start,
                                     end=start + len(token),
                                     sent=row))

        return rows

    def _random_sampling(self, dataset: list, n: int) -> list:
        """
        A private method that randomly samples n number of sentences containing target words from the given dataset.

        Args:
            dataset(list): A list of strings representing the dataset to be sampled.
            n(int): An integer representing the number of sentences to be sampled.

        Returns:
            dataset(list)
        """
        dataset = Dataset.from_dict({'text': dataset})
        dataset.shuffle(seed=SEED)
        return dataset['text'][:n]


class SpacyTokenization(Tokenization, ABC):
    def __init__(self, spacy_model: str, words: list):
        """
        Args:
            spacy_model(str): A string representing the name of the spaCy model to be used.
            words(list): A list of target words.
        """
        super().__init__(words)
        set_seed(SEED)

        self._nlp = spacy.load(spacy_model, disable=['parser', 'ner', 'senter'])

    def tokenize(self, dataset: str, sampling: int = 0) -> list:
        # load dataset
        dataset = load_dataset('text', data_files=dataset, split='train')
        dataset = dataset.map(lambda batch: {'text': [text.replace('_nn', '').replace('_vb', '')
                                                      for text in batch['text']]}, batched=True)['text']

        # random sampling
        if sampling:
            dataset = self._random_sampling(dataset, sampling)

        # spacy tokenization
        docs = list(self._nlp.pipe(dataset, n_process=-1, batch_size=1000))

        # collect data
        rows = list()
        for idx, row in enumerate(docs):
            for token in row:
                if self.is_doc_relevant(row, token):
                    rows.append(dict(sentidx=idx,
                                     lemma=token.lemma_,
                                     token=token.text,
                                     start=token.idx,
                                     end=token.idx + len(token.text),
                                     sent=str(row)))

        return rows


class StandardSpacyTokenization(SpacyTokenization):
    """
    This class tokenizes text using a given spaCy model and a list of target words.
    Only the sentences containing the lemma of target words will be considered for tokenization.
    """

    def is_doc_relevant(self, row: Doc, token: Token) -> bool:
        return token.lemma_ in self.words

class LatinSpacyTokenization(SpacyTokenization):
    """
    This class tokenizes text using a given spaCy model and a list of target words.
    Only the sentences containing the (lemma of) target words will be considered for tokenization.

    Spacy Lemmatizer for Italian is not so good.
    """

    def is_doc_relevant(self, row: Doc, token: Token) -> bool:
        return token.lemma_ in self.words or token.text.lower() in self.words
        
class ItalianSpacyTokenization(SpacyTokenization):
    """
    This class tokenizes text using a given spaCy model and a list of target words.
    Only the sentences containing the (lemma of) target words will be considered for tokenization.

    Spacy Lemmatizer for Italian is not so good.
    """

    def is_doc_relevant(self, row: Doc, token: Token) -> bool:
        return token.lemma_ in self.words or token.text.lower() in self.words


class StandardTokenization(Tokenization):
    """
    This class tokenizes text using a given spaCy model and a list of target words.
    Only the sentences containing the target words will be considered for tokenization.
    """

    def is_doc_relevant(self, row: object, token: str) -> bool:
        return token in self.words

if __name__ == '__main__':
    import argparse
    import importlib
    import pandas as pd
    from pathlib import Path

    parser = argparse.ArgumentParser(prog='Tokenization script for processing LSC benchmark datasets.',
                                     add_help=True)
    parser.add_argument('-d', '--dataset',
                        type=str,
                        help='Path to the directory containing the benchmark dataset for LSC detection. '
                             'This directory should include a file named \'targets.txt\' which contains a list of target words, and subdirectories for each corpus to be tokenized.')
    parser.add_argument('-m', '--model',
                        type=str,
                        help='Name of the spaCy model to use for tokenization.')
    parser.add_argument('-n', '--sampling',
                        type=int,default=0,
                        help='Number of sentences to randomly sample from the dataset. '
                             'Default value is 0, which means no sampling will be performed.')
    parser.add_argument('-o', '--output',
                        type=str,
                        help='Path to the output directory where the selected sentences will be stored. '
                             'This directory will be created if it doesn\'t exist.')
    parser.add_argument('-t', '--tokenization_class',
                        type=str, default='StandardSpacyTokenization',
                        help='Name of the Tokenization class to use for tokenization. '
                             'This should be the name of a class that extends the abstract base class Tokenization in the src.tokenization module.')
    args = parser.parse_args()

    # reflection -> get the class to instanziate
    module = importlib.import_module(__name__)
    tokenization_class = getattr(module, args.tokenization_class)

    # target words
    # target words
    targets_filename = f'{args.dataset}/targets.txt' if not 'Russian' in args.dataset else f'{args.dataset}/targets_test.txt'
    words = [word.strip() for word in open(targets_filename, mode='r', encoding='utf-8').readlines()]

    # Tokenize the raw corpora
    if args.model is not None:
         tokenizer = tokenization_class(args.model, words)
    else:
         tokenizer = tokenization_class(words)

    for corpus in ['corpus1', 'corpus2']:
        dataset_input = f'{args.dataset}/{corpus}/token/{corpus}.txt'
        tokenization_output = f'{args.output}/{corpus}/token'

        # run tokenization
        token_list = tokenizer.tokenize(dataset_input, args.sampling)

        # store results
        Path(tokenization_output).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(token_list)
        for word in words:
            filename = f'{tokenization_output}/{word}.txt'
            df[df['lemma'] == word].to_json(filename, orient='records', lines=True)
