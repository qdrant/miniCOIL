import argparse
from collections import defaultdict
from typing import List
import tqdm
import json

from py_rust_stemmers import SnowballStemmer
from nltk import WordNetLemmatizer


def read_source_vocab(path) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    args = parser.parse_args()

    import nltk

    nltk.download('wordnet')

    source_vocab = read_source_vocab(args.input_file)

    # Remove stopwords and words with less than 3 characters
    # As vocab is sorted by frequency, define stopwords as first 100 words if the word is less than 5

    target_vocab = [word for idx, word in enumerate(source_vocab) if len(word) > (5 if idx < 100 else 2)]

    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    normalized_words = defaultdict(set)

    for word in tqdm.tqdm(target_vocab):
        anchor_word = stemmer.stem_word(word)
        lemmatized_word = lemmatizer.lemmatize(word)

        normalized_words[anchor_word].add(word)
        normalized_words[anchor_word].add(lemmatized_word)

    normalized_words = {k: list(v) for k, v in normalized_words.items()}

    with open(args.output_file, "w") as f:
        json.dump(normalized_words, f, indent=4)


if __name__ == "__main__":
    main()
