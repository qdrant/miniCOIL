"""
This script converts the OpenWebText dataset to the usable format.

Default format, apparently, was created by data scientists, who have no idea how to properly store data.
So they did this:

Main file (tar)
    - folder
        - sub-archive file (xz compressed)
            - tar file
                - text file
                - text file
                - text file

This script will convert all this nonsense to the simple compressed texts file,
which is possible to decompress on the fly with the CLI tools any teapot can run.
"""

import os
import tarfile
import glob
import tqdm
import gzip
import hashlib
from typing import Iterable

from sentence_splitter import SentenceSplitter

from mini_coil.settings import DATA_DIR


def iterate_archives() -> Iterable[str]:
    path_to_xz_archives = os.path.join(DATA_DIR, "openwebtext", "*.xz")
    all_files = glob.glob(path_to_xz_archives)
    for archive in tqdm.tqdm(all_files):
        yield archive


def read_files_from_tar_xz(archive_path: str) -> Iterable[str]:
    with tarfile.open(archive_path, "r:xz") as tar:
        for member in tar.getmembers():
            yield member.name, tar.extractfile(member).read().decode("utf-8")


def read_texts() -> Iterable[str]:
    for archive in iterate_archives():
        for file_name, text in read_files_from_tar_xz(archive):
            file_hash = compute_hash(file_name)
            yield file_hash, text


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    output_file = os.path.join(DATA_DIR, "openwebtext.txt.gz")

    splitter = SentenceSplitter(language='en')

    with gzip.open(output_file, "wt") as f:
        for file_hash, text in tqdm.tqdm(read_texts()):
            for sentence in splitter.split(text):
                    sentence = sentence.strip()
                    if len(sentence) > 0:
                        f.write(f"{file_hash}\t{sentence}\n")


if __name__ == "__main__":
    main()
