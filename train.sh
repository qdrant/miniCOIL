#! /usr/bin/env bash

set -e # exit on error
set -u # exit on using unset variable
set -o pipefail # exit on error in pipe

CURRENT_DIR=$(pwd -L)
#COLLECTION_NAME=$1
TARGET_WORD=bat
DIM=4
SAMPLES=4000
NEIGHBORS=20
LMODEL=mxbai-large

. venv/bin/activate || . .venv/bin/activate || true

## Convert dataset into readable format

## Split data into sentences
#python -m mini_coil.data_pipeline.split_sentences \
#  --input-file "${CURRENT_DIR}/data/openwebtext-1920-sentences-"${TARGET_WORD}".txt.gz" \
#  --output-file "${CURRENT_DIR}/data/openwebtext-sentences/openwebtext-1920-splitted-"${TARGET_WORD}".txt.gz"

## Encode sentences with transformer model
#python -m  mini_coil.data_pipeline.encode_targets \
#  --input-file "${CURRENT_DIR}/data/openwebtext-1920-sentences-"${TARGET_WORD}".txt.gz" \
#  --output-file "${CURRENT_DIR}/data/output/openwebtext-1920-splitted-"${TARGET_WORD}"-encodings"

## Upload encoded sentences to Qdrant
#python -m mini_coil.data_pipeline.upload_to_qdrant \
#  --input-emb
#  --input-text
#  --collection-name ${COLLECTION_NAME}

# Sample sentences with specified words and apply dimensionality reduction
#python -m mini_coil.data_pipeline.compress_dimentions \
#  --output-dir data/umap-"${SAMPLES}"-"${NEIGHBORS}"-"${DIM}"d-"${LMODEL}" \
#  --sample-size "${SAMPLES}" --dim "${DIM}" --word "${TARGET_WORD}" --overwrite \
#  --limit ${NEIGHBORS} --n_neighbours ${NEIGHBORS}

echo "Compressed dimentions"
## Download sampled sentences
python -m mini_coil.data_pipeline.load_sentences \
   --word "${TARGET_WORD}" \
   --matrix-dir data/umap-"${SAMPLES}"-"${NEIGHBORS}"-"${DIM}"d-"${LMODEL}" \
   --output-dir data/umap-"${SAMPLES}"-"${NEIGHBORS}"-"${DIM}"d-"${LMODEL}"-sentences

echo "Loaded sentences"
## Encode sentences with smaller transformer model
python -m mini_coil.data_pipeline.encode_and_filter \
   --sentences-file data/umap-"${SAMPLES}"-"${NEIGHBORS}"-"${DIM}"d-"${LMODEL}"-sentences/sentences-"${TARGET_WORD}".jsonl \
   --output-file data/umap-"${SAMPLES}"-"${NEIGHBORS}"-"${DIM}"d-"${LMODEL}"-input/word-emb-"${TARGET_WORD}".npy \
   --word "${TARGET_WORD}" \
   --sample-size "${SAMPLES}"

echo "Encoded sentences"
#Train encoder **for each word**
 python -m mini_coil.training.train_word \
  --embedding-path data/umap-"${SAMPLES}"-"${NEIGHBORS}"-"${DIM}"d-"${LMODEL}"-input/word-emb-"${TARGET_WORD}".npy \
  --target-path data/umap-"${SAMPLES}"-"${NEIGHBORS}"-"${DIM}"d-"${LMODEL}"/compressed_matrix_"${TARGET_WORD}".npy \
  --log-dir data/train_logs/log_"${TARGET_WORD}" \
  --output-path data/umap-"${SAMPLES}"-"${NEIGHBORS}"-"${DIM}"d-"${LMODEL}"-models/model-"${TARGET_WORD}".ptch \
  --epochs 500
##  --gpu

echo "Combined models"
## Merge encoders for each word into a single model
python -m mini_coil.data_pipeline.combine_models \
  --models-dir "${CURRENT_DIR}/data/umap-${SAMPLES}-${NEIGHBORS}-${DIM}d-${LMODEL}-models" \
  --vocab-path "${CURRENT_DIR}/data/30k-vocab-filtered.txt" \
  --output-path "data/model_${SAMPLES}_${DIM}d" \
  --output-dim "${DIM}"

