import re
import abc
import json
import torch
import random
import numpy as np
from datasets import Dataset, logging as dataset_logging
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging

# avoid boring logging
dataset_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# The Answer to the Great Question of Life, the Universe and Everything is Forty-two
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Extraction(abc.ABC):
    """
    An abstract base class for extracting data from a pre-trained BERT-based model.
    """

    def __init__(self, pretrained: str = 'bert-base-uncased',
                 output_hidden_states: bool = False,
                 output_attentions: bool = False):
        """
        Initializes a pre-trained BERT-based model.

        Args:
            pretrained (str, default='bert-base-uncased'): The pre-trained model name or path to load.
            output_hidden_states (bool, default=False): Whether to output all hidden-states of the model.
            output_attentions (bool, default=False): Whether to output attentions weights of the model.
        """

        self._device = self._set_seed_and_device()

        # load hugginface tokenizer and model
        self.pretrained = pretrained
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, strip_accents=True)
        self.model = AutoModel.from_pretrained(pretrained,
                                               output_hidden_states=output_hidden_states,
                                               output_attentions=output_attentions)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        _ = self.model.to(self._device)
        _ = self.model.eval()
        self.split_token = '_' if pretrained == 'xlm-roberta-base' else '#'*2

    def _set_seed_and_device(self) -> object:
        """
        Determines the device (CPU or GPU) to run the model on.

        Returns:
            object, a torch device object.
        """
        set_seed(SEED)
        self._device_name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(self._device_name)

    def _tokenize_factory(self, tokenizer: object, max_length: int = None) -> object:
        """
        A factory function that returns a function used to tokenize text using the provided tokenizer.

        Args:
            tokenizer(object): The tokenizer object used for tokenization
            max_length(int, optional): The maximum sequence length to use. If None, the max_length possibile is used.

        Returns:
            function
        """
        max_length = max_length if max_length is not None else tokenizer.model_max_length

        def tokenize(examples) -> dict:
            """Tokenization function"""
            return tokenizer(examples["sent"], return_tensors='pt',
                             padding="max_length", max_length=max_length, truncation=True).to(self._device)

        return tokenize

    def _tokens2str(self, tokens: list, tokenizer: object) -> str:
        """
        Convert a list of tokens to a string representation. This function takes in a list of tokens and returns a string representation of the tokens, where each token is separated by a space.

        Args:
            tokens(list): The list of tokens to be converted to a string.
            tokenizer(object): The BertTokenizer used for training the model.

        Returns:
            a string representation of the input list of tokens.
        """
        return " ".join(tokenizer.convert_ids_to_tokens(tokens))

    def _load_dataset(self, dataset: str) -> Dataset:
        """
        This method loads a dataset from a file and returns it as a Dataset object.

        Args:
            dataset (str): A string representing the path to the dataset file.

        Returns:
            A Dataset object representing the loaded dataset.
        """
        rows = list()
        with open(dataset, mode='r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == '': continue
                row = json.loads(line)
                if row is None: continue
                rows.append(row)

        #return Dataset.from_pandas(pd.DataFrame(rows))
        return Dataset.from_list(rows)

    def _tokenize_dataset(self, dataset: Dataset, max_length: int) -> Dataset:
        """
        Tokenizes the text data in a dataset using the BERT tokenizer.

        Args:
            dataset (Dataset): A Dataset object containing the text data to be tokenized.
            max_length (int): The maximum length of the sequences after tokenization.

        Returns:
            A Dataset object containing the tokenized text data.
        """
        tokenize_func = self._tokenize_factory(self.tokenizer, max_length)
        dataset = dataset.map(tokenize_func, batched=True)
        dataset.set_format('torch')
        return dataset

    @abc.abstractmethod
    def extract(self, dataset: str, batch_size:int=8, max_length:int=512, layers:int=12, sampling:int=0, **kwargs) -> object:
        """
        Abstract method for extracting embeddings or attention from a pre-trained BERT-based model.

        Args:
            dataset (str): The path to the dataset directory containing the sentences to extract embeddings/attention from.
            batch_size (int, default=8): The batch size to use for extracting the embeddings/attention.
            max_length (int, default=512): The maximum sequence length to use for tokenization.
            layers (int, default=12): The number of layers in the pre-trained model to extract embeddings/attention from.
            **kwargs: Additional arguments specific to the subclass implementing this method.

        Returns:
            A dict or list representing the extracted embeddings or attention, respectively.
        """


class WordEmbeddingExtraction(Extraction):
    def __init__(self, pretrained: str = 'bert-base-uncased'):
        super().__init__(pretrained, output_hidden_states=True, output_attentions=False)

    def extract(self,
                dataset: str,
                batch_size:int=8,
                max_length:int=512,
                layers:int=12,
                sampling:int=0,
                agg_sub_words:str='mean') -> dict:
        """
        Extracts word embeddings for a list of target words.

        Args:
            dataset (str): The path to the dataset directory containing the sentences to extract embeddings from.
            batch_size (int, default=8): The batch size to use for extracting the embeddings.
            max_length (int, default=512): The maximum sequence length to use for tokenization.
            layers (int, default=12): The number of layers in the pre-trained model to extract embeddings from.
            sampling (int, default=0): The number of examples to sample from the dataset. If set to 0, all examples in the dataset are used.
            agg_sub_words (str, default='mean'): The aggregation function to use to obtain the embedding for sub-words.

        Returns:
            A dict or tensor representing the extracted embeddings.
        """

        # load dataset
        dataset = self._load_dataset(dataset)

        text = dataset.select_columns('sent')
        offset = dataset.remove_columns('sent')

        # tokenize text
        tokenized_text = self._tokenize_dataset(text, max_length)

        # collect embedding to store on disk
        embeddings = dict()
        for i in range(0, tokenized_text.shape[0], batch_size):
            start, end = i, min(i + batch_size, text.num_rows)
            batch_offset = offset.select(range(start, end))
            batch_text = text.select(range(start, end))
            batch_tokenized_text = tokenized_text.select(range(start, end))

            # to device
            input_ids = batch_tokenized_text['input_ids'].to(self._device)
            if 'token_type_ids' in batch_tokenized_text:
                token_type_ids = batch_tokenized_text['token_type_ids'].to(self._device)
            else:
                token_type_ids = None
            attention_mask = batch_tokenized_text['attention_mask'].to(self._device)

            # model prediction
            with torch.no_grad():
                if token_type_ids is None:
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    output = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)

            # hidden states
            hidden_states = torch.stack(output['hidden_states'])

            # select the embeddings of a specific target word
            for j, row in enumerate(batch_tokenized_text):
                input_tokens_str = self._tokens2str(row['input_ids'].tolist(), self.tokenizer)
                word_tokens = batch_text[j]['sent'][batch_offset[j]['start']:batch_offset[j]['end']]
                word_tokens_str = " ".join(self.tokenizer.tokenize(word_tokens))

                try:
                    pos = re.search(f"( +|^){word_tokens_str}(?!\w+| {self.split_token})", input_tokens_str, re.DOTALL)
                except:
                    print('--\n', f"( +|^){word_tokens_str}(?!\w+| {self.split_token})", '\n', input_tokens_str)
                    continue

                # truncation side effect
                if pos is None:
                    idx_original_sent=j+i*batch_tokenized_text.num_rows
                    print(f"Pretrained: {self.pretrained},\n, {type(self.tokenizer)}\n, {word_tokens}\n, {batch_text[j]}\n, {batch_offset[j]['start']}:{batch_offset[j]['end']}, idx: {idx_original_sent},\nword_tokens_str: {word_tokens_str},\ninput_tokens_str: {input_tokens_str}")
                    continue

                pos = pos.start()
                n_previous_tokens = len(input_tokens_str[:pos].split())
                n_word_token = len(word_tokens_str.split())

                # get the embeddings from each layer
                for l in range(1, layers + 1):
                    sub_word_state = hidden_states[l, j][n_previous_tokens: n_previous_tokens + n_word_token]
                    word_state = torch.__dict__[agg_sub_words](sub_word_state, dim=0).unsqueeze(0)
                    if l in embeddings:
                        embeddings[l] = torch.vstack([embeddings[l], word_state])
                    else:
                        embeddings[l] = word_state

        return embeddings