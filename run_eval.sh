#! /usr/bin/env bash

set -e # exit on error
set -u # exit on using unset variable
set -o pipefail # exit on error in pipe

mkdir -p tests/em/t || true
CURRENT_DIR=$(pwd -L)
DIM=4

WORD_TO_TEST=vector
MINICOIL_MODEL=triplets-8000-mxbai-large-jina-small-${DIM}-augmented/full-models/model

SENTENCES_FILE="${CURRENT_DIR}/data/validation/${WORD_TO_TEST}-validation.txt"


function create_embeddings() {
  echo "Creating embeddings"
  python -m tests.01_embedding_maker \
    --minicoil-test-word "${WORD_TO_TEST}" \
    --input-file "${SENTENCES_FILE}" \
    --vocab-path "${CURRENT_DIR}/data/${MINICOIL_MODEL}.vocab" \
    --word-encoder-path "${CURRENT_DIR}/data/${MINICOIL_MODEL}.npy" \
    --output-random "${CURRENT_DIR}/tests/em/random.npy" \
    --output-mixedbread "${CURRENT_DIR}/tests/em/mixedbread.npy" \
    --output-jina "${CURRENT_DIR}/tests/em/jina_small.npy" \
    --output-minicoil "${CURRENT_DIR}/tests/em/minicoil.npy" \
    --dim "${DIM}"
}

function create_matrix() {
    echo "Creating distance matrix: Minicoil"
    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/minicoil.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_minicoil.npy"

    echo "Creating distance matrix: Mixedbread"
    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/mixedbread.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy"

    echo "Creating distance matrix: Jina"
    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/jina_small.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_jina_small.npy"

    echo "Creating distance matrix: Random"
    python -m tests.02_matrix_create \
        --input "${CURRENT_DIR}/tests/em/random.npy" \
        --output "${CURRENT_DIR}/tests/em/distance_matrix_random.npy"
}

function evaluate() {
    echo "Evaluating"
    python -m tests.03_matrix_triplets \
      --distance-matrix-base-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy" \
      --distance-matrix-eval-path "${CURRENT_DIR}/tests/em/distance_matrix_minicoil.npy" \
      --sample-size 10000 \
      --base-margin 0.1 \
      --eval-margin 0.01

    python -m tests.03_matrix_triplets \
      --distance-matrix-base-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy" \
      --distance-matrix-eval-path "${CURRENT_DIR}/tests/em/distance_matrix_random.npy" \
      --sample-size 10000 \
      --base-margin 0.0000123 \
      --eval-margin 0.0000123
}





create_embeddings
create_matrix
evaluate






####################################
# UMAP COMPARE (disabled by default)

function umap_compare() {
    python -m tests.04_umap_emb \
      --embeddings "${CURRENT_DIR}/tests/em/mixedbread.npy" \
      --output-umap "${CURRENT_DIR}/tests/em/mixedbread_umap.npy" \
      --umap-components 4 \
      --n-neighbors 20

    python -m tests.02_matrix_create \
            --input "${CURRENT_DIR}/tests/em/mixedbread_umap.npy" \
            --output "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread_umap.npy"

    python -m tests.03_matrix_triplets \
      --distance-matrix-base-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread.npy" \
      --distance-matrix-eval-path "${CURRENT_DIR}/tests/em/distance_matrix_mixedbread_umap.npy" \
      --sample-size 100000 \
      --base-margin 0.1 \
      --eval-margin 0.01
}

#umap_compare
