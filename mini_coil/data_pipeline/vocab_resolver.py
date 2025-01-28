from collections import defaultdict
from typing import Iterable, Tuple, List

from py_rust_stemmers import SnowballStemmer

import numpy as np
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from mini_coil.data_pipeline.stopwords import english_stopwords


class VocabTokenizer:

    def tokenize(self, sentence: str) -> np.ndarray:
        raise NotImplementedError()

    def convert_ids_to_tokens(self, token_ids: np.ndarray) -> list:
        raise NotImplementedError()


class VocabTokenizerAutoTokenizer(VocabTokenizer):
    def __init__(self, model_repository: str):
        self.auto_tokenizer = AutoTokenizer.from_pretrained(model_repository)

    def tokenize(self, sentence: str) -> np.ndarray:
        return np.array(self.auto_tokenizer(sentence).input_ids)

    def convert_ids_to_tokens(self, token_ids: np.ndarray) -> list:
        return self.auto_tokenizer.convert_ids_to_tokens(token_ids)


class VocabTokenizerTokenizer(VocabTokenizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, sentence: str) -> np.ndarray:
        return np.array(self.tokenizer.encode(sentence).ids)

    def convert_ids_to_tokens(self, token_ids: np.ndarray) -> list:
        return [self.tokenizer.id_to_token(token_id) for token_id in token_ids]


class VocabResolver:
    def __init__(self, model_repository: str = None, tokenizer: VocabTokenizer = None):
        # Word to id mapping
        self.vocab = {}
        # Id to word mapping
        self.words = []
        # Lemma to word mapping
        self.stem_mapping = {}
        self.tokenizer: VocabTokenizer = tokenizer
        self.stemmer = SnowballStemmer("english")
        if model_repository is not None and tokenizer is None:
            self.tokenizer = VocabTokenizerAutoTokenizer(model_repository)

    def tokenize(self, sentence: str) -> np.ndarray:
        return self.tokenizer.tokenize(sentence)

    def lookup_word(self, word_id: int) -> str:
        if word_id == 0:
            return "UNK"
        return self.words[word_id - 1]

    def convert_ids_to_tokens(self, token_ids: np.ndarray) -> list:
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def vocab_size(self):
        return len(self.vocab) + 1

    def save_vocab(self, path):
        with open(path, "w") as f:
            for word in self.words:
                f.write(word + "\n")

    def save_json_vocab(self, path):
        import json
        with open(path, "w") as f:
            json.dump({
                "vocab": self.words,
                "stem_mapping": self.stem_mapping
            }, f, indent=2)

    def load_json_vocab(self, path):
        import json
        with open(path, "r") as f:
            data = json.load(f)
            self.words = data["vocab"]
            self.vocab = {word: idx + 1 for idx, word in enumerate(self.words)}
            self.stem_mapping = data["stem_mapping"]


    def add_word(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab) + 1
            self.words.append(word)
            stem = self.stemmer.stem_word(word)
            if stem not in self.stem_mapping:
                self.stem_mapping[stem] = word
            else:
                existing_word = self.stem_mapping[stem]
                if len(existing_word) > len(word):
                    # Prefer shorter words for the same stem
                    # Example: "swim" is preferred over "swimming"
                    self.stem_mapping[stem] = word

    def load_vocab(self, path):
        with open(path, "r") as f:
            for line in f:
                self.add_word(line.strip())

    @classmethod
    def _reconstruct_bpe(
            self, bpe_tokens: Iterable[Tuple[int, str]]
    ) -> List[Tuple[str, List[int]]]:
        result = []
        acc = ""
        acc_idx = []

        continuing_subword_prefix = "##"
        continuing_subword_prefix_len = len(continuing_subword_prefix)

        for idx, token in bpe_tokens:

            if token.startswith(continuing_subword_prefix):
                acc += token[continuing_subword_prefix_len:]
                acc_idx.append(idx)
            else:
                if acc:
                    result.append((acc, acc_idx))
                    acc_idx = []
                acc = token
                acc_idx.append(idx)

        if acc:
            result.append((acc, acc_idx))

        return result

    def resolve_tokens(self, token_ids: np.ndarray) -> (np.ndarray, dict, dict, dict):
        """
        Mark known tokens (including composed tokens) with vocab ids.

        Args:
            token_ids: (seq_len) - list of ids of tokens
                Example:
                    [
                        101,  3897, 19332, 12718, 23348,
                        1010,  1996,  7151,  2296, 4845,
                        2359,  2005,  4234,  1010,  4332,
                        2871,  3191,  2062, 102
                    ]

            returns:
                - token_ids with vocab ids
                    [
                        0,  151, 151, 0, 0,
                        912,  0,  0,  0, 332,
                        332,  332,  0,  7121,  191,
                        0,  0,  332, 0
                    ]
                - counts of each token
                    {
                        151: 1,
                        332: 3,
                        7121: 1,
                        191: 1,
                        912: 1
                    }
                - oov counts of each token
                    {
                        "the": 1,
                        "a": 1,
                        "[CLS]": 1,
                        "[SEP]": 1,
                        ...
                    }
                - forms of each token
                    {
                        "hello": ["hello"],
                        "world": ["worlds", "world", "worlding"],
                    }

        """

        # deep copy of token_ids
        token_ids = token_ids.copy()
        tokens = self.convert_ids_to_tokens(token_ids)
        tokens_mapping = self._reconstruct_bpe(enumerate(tokens))

        counts = defaultdict(int)
        oov_count = defaultdict(int)

        forms = defaultdict(list)

        for token, mapped_token_ids in tokens_mapping:
            vocab_id = 0
            if token in english_stopwords:
                vocab_id = 0
            elif token in self.vocab:
                vocab_id = self.vocab[token]
                forms[token].append(token)
            elif token in self.stem_mapping:
                vocab_id = self.vocab[self.stem_mapping[token]]
                forms[self.stem_mapping[token]].append(token)
            else:
                stem = self.stemmer.stem_word(token)
                if stem in self.stem_mapping:
                    vocab_id = self.vocab[self.stem_mapping[stem]]
                    forms[self.stem_mapping[stem]].append(token)

            for token_id in mapped_token_ids:
                token_ids[token_id] = vocab_id

            if vocab_id == 0:
                oov_count[token] += 1
            else:
                counts[vocab_id] += 1

        return token_ids, counts, oov_count, forms

    def token_ids_to_vocab_batch(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Mark known tokens (including composed tokens) with vocab ids.

        Args:
            token_ids: (batch_size, seq_len) - list of ids of tokens
                Example:
                    [
                        [101,  3897, 19332, 12718, 23348],
                        [1010,  1996,  7151,  2296, 4845],
                        [2359,  2005,  4234,  1010,  4332],
                        [2871,  3191,  2062, 102, 0]
                    ]

        """

        for i in range(token_ids.shape[0]):
            self.resolve_tokens(token_ids[i])

        return token_ids

    def filter(
            self,
            token_ids: np.ndarray,
            token_embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter out tokens that are not in the vocab.

        Args:
            token_ids: (batch_size, seq_len) - list of ids of tokens
            token_embeddings: (batch_size, seq_len, embedding_size) - embeddings of tokens

        Returns:
            - number of tokens in each sequence - (batch_size)
            - filtered and flattened token_ids - (total_tokens_size)
            - filtered and flattened token_embeddings - (total_tokens_size, embedding_size)
        """

        # (batch_size, seq_len)
        filtered_token_ids = self.token_ids_to_vocab_batch(token_ids)

        # (batch_size, seq_len)
        mask = filtered_token_ids.__ne__(0)

        # (batch_size)
        num_tokens = mask.sum(axis=1)

        # (total_tokens_size)
        filtered_token_ids = filtered_token_ids[mask]

        # (total_tokens_size, embedding_size)
        filtered_token_embeddings = token_embeddings[mask]

        return num_tokens, filtered_token_ids, filtered_token_embeddings


def test_basic_resolver():
    resolver = VocabResolver()

    resolver.add_word("bat")
    resolver.add_word("nicolls")

    token_ids = np.array([
        101, 3897, 19332, 12718, 23348,
        1010, 1996, 7151, 2296, 4845,
        2359, 2005, 4234, 1010, 4332,
        2871, 3191, 2062, 102
    ])

    token_ids, counts, oov, _forms = resolver.resolve_tokens(token_ids)

    expected = np.array([0, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    assert np.all(np.equal(token_ids, expected))

    batch = np.array([
        [101, 3897, 19332, 12718, 23348],
        [1010, 1996, 7151, 2296, 4845],
        [2359, 2005, 4234, 1010, 4332],
        [2871, 3191, 2062, 102, 0]
    ])

    batch = resolver.token_ids_to_vocab_batch(batch)

    expected = np.array([
        [0, 0, 2, 2, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    assert np.all(np.equal(batch, expected))


def main():
    import os
    from mini_coil.settings import DATA_DIR

    resolver = VocabResolver(model_repository="jinaai/jina-embeddings-v2-small-en")

    resolver.load_json_vocab(os.path.join(DATA_DIR, "minicoil.ptch.vocab"))

    sentence = "I like to swim close to the bank of the river, cause I am not a very good swimmer. He swims slow."

    token_ids = np.array(resolver.tokenizer.tokenize(sentence))

    word_ids, counts, oov, forms = resolver.resolve_tokens(token_ids)

    print("word_ids", word_ids)

    print("counts", counts)

    print("oov", oov)

    print("forms", forms)


if __name__ == "__main__":
    main()
