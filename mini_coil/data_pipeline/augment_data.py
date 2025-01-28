import argparse
import json
import logging
import os
import random
import re
from typing import Tuple, List

from mini_coil.settings import DATA_DIR

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logging.getLogger("data-augment").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def extract_window_around_target(
        words: List[str],
        target_idx: int,
        min_words: int = 4,
        max_words: int = 6
) -> Tuple[int, int]:
    left_window = random.randint(1, 3)
    right_window = random.randint(1, 3)

    start = max(0, target_idx - left_window)
    end = min(len(words), target_idx + right_window + 1)

    while end - start < min_words:
        if end < len(words):
            end += 1
        elif start > 0:
            start -= 1
        else:
            break

    if end - start > max_words:
        if random.choice([True, False]):
            end = start + max_words
        else:
            start = end - max_words

    return start, end


def create_snippet(sentence: str, target_word_forms: List[str]) -> str:
    words = sentence.split()

    if not words:
        return sentence

    normalized_target_forms = set(
        re.sub(r"\W+", "", target_word.lower())
        for target_word in target_word_forms
    )

    try:
        target_indices = [
            i for i, w in enumerate(words)
            if re.sub(r"\W+", "", w.lower()) in normalized_target_forms
        ]

        if not target_indices:
            return sentence

        target_idx = random.choice(target_indices)
        start, end = extract_window_around_target(words, target_idx)

        return " ".join(words[start:end])

    except Exception as e:
        logger.exception("Error creating snippet: ", e)
        return sentence


def process_file(input_path: str, output_path: str, target_word_forms: List[str]) -> None:
    try:
        with open(input_path, "r", encoding="utf-8") as fin, \
                open(output_path, "w", encoding="utf-8") as fout:

            for i, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    data["line_number"] = i

                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                    data_copy = dict(data)
                    data_copy["sentence"] = create_snippet(data["sentence"], target_word_forms)

                    if data_copy["sentence"] != data["sentence"]:
                        fout.write(json.dumps(data_copy, ensure_ascii=False) + "\n")

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue

    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")


def main():
    default_vocab_path = os.path.join(DATA_DIR, "30k-vocab-filtered.json")

    parser = argparse.ArgumentParser(description="Create snippets around target words in sentences")
    parser.add_argument("--input-file", required=True, help="Path to input .jsonl file")
    parser.add_argument("--output-file", required=True, help="Path to output .jsonl file")
    parser.add_argument("--target-word", required=True, help="Target word to create snippet around")
    parser.add_argument("--vocab-path", type=str, default=default_vocab_path)
    args = parser.parse_args()

    vocab = json.load(open(args.vocab_path))

    if args.target_word not in vocab:
        print(f"WARNING: word {args.target_word} not found in vocab, using as is")
        word_forms = [args.target_word]
    else:
        word_forms = vocab[args.target_word]

    process_file(args.input_file, args.output_file, word_forms)


if __name__ == "__main__":
    main()
