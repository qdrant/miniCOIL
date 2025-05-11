#! /usr/bin/env bash

set -e # exit on error
set -u # exit on using unset variable
set -o pipefail # exit on error in pipe

CURRENT_DIR=$(pwd -L)
#COLLECTION_NAME=$1
TARGET_WORD=${1:-"calcul"}
DIM=4
SAMPLES=8000
IMODEL=jina-small


INPUT_DIR=data/parametric-umap-${SAMPLES}

MODEL_DIR=${INPUT_DIR}-${IMODEL}-${DIM}-augmented

WORD_MODELS_DIR=${MODEL_DIR}/word-models
FULL_MODEL_DIR=${MODEL_DIR}/full-models


sample_data() {
  python -m mini_coil.data_pipeline.sample_data \
    --output-file ${INPUT_DIR}/target_sentences/sentences-${TARGET_WORD}.jsonl \
    --sample-size ${SAMPLES}
}


augment_data() {
  python -m mini_coil.data_pipeline.augment_data \
    --input-file ${INPUT_DIR}/target_sentences/sentences-${TARGET_WORD}.jsonl \
    --output-file ${INPUT_DIR}/target_sentences/sentences-${TARGET_WORD}-augmented.jsonl \
    --target-word "${TARGET_WORD}"
  echo "Augmented data"
}


encode_sentences() {
  # Encode sentences with smaller transformer model
  python -m mini_coil.data_pipeline.encode_and_filter \
     --sentences-file ${INPUT_DIR}/target_sentences/sentences-${TARGET_WORD}-augmented.jsonl \
     --output-file ${INPUT_DIR}-${IMODEL}/word-emb-${TARGET_WORD}.npy \
     --output-line-numbers-file ${INPUT_DIR}-${IMODEL}/line-numbers-${TARGET_WORD}.npy \
     --word "${TARGET_WORD}"
  echo "Encoded sentences"
}


train_encoder() {
  #Train encoder **for each word**
  CUDA_VISIBLE_DEVICES=-1 python -m mini_coil.training.train_word_parametric_umap \
    --embedding-path ${INPUT_DIR}-${IMODEL}/word-emb-${TARGET_WORD}.npy \
    --output-dim ${DIM} \
    --output-model-path ${MODEL_DIR}/model-${TARGET_WORD}.npy \
    --output-embeddings-path ${MODEL_DIR}/embeddings-${TARGET_WORD}.npy
  echo "Trained model: ${MODEL_DIR}/model-${TARGET_WORD}.npy"
}


combine_models() {
  ## Merge encoders for each word into a single model
  python -m mini_coil.data_pipeline.combine_parametric_umap_models \
    --models-dir ${MODEL_DIR} \
    --vocab-path "${CURRENT_DIR}/data/30k-vocab-filtered.json" \
    --output-path ${FULL_MODEL_DIR}/model \
    --output-dim "${DIM}"
}


download_validation_data() {
  # Download validation data
  python -m mini_coil.data_pipeline.download_validation \
    --word ${TARGET_WORD} \
    --output-sentences data/validation/${TARGET_WORD}-validation.txt
}


embed_sentences() {
  # Embed a bunch of sentences
  python -m tests.embed_minicoil \
    --vocab-path ${FULL_MODEL_DIR}/model.vocab \
    --word-encoder-path ${FULL_MODEL_DIR}/model.npz \
    --input-file data/validation/${TARGET_WORD}-validation.txt \
    --word ${TARGET_WORD} \
    --output ${WORD_MODELS_DIR}/validation/${TARGET_WORD}-validation.npy
}

visualize_embeddings() {
  # Plot the embeddings
  python -m tests.visualize_embeddings \
    --input ${WORD_MODELS_DIR}/validation/${TARGET_WORD}-validation.npy \
    --output ${WORD_MODELS_DIR}/validation-viz/${TARGET_WORD}-plot
}

cleaenup() {
  rm ${INPUT_DIR}/distance_matrix/dm-${TARGET_WORD}.npy
  rm ${INPUT_DIR}/target_sentences/sentences-${TARGET_WORD}.jsonl
  rm ${INPUT_DIR}/target_sentences/sentences-${TARGET_WORD}-augmented.jsonl
  rm ${INPUT_DIR}-${IMODEL}/word-emb-${TARGET_WORD}.npy
  rm ${INPUT_DIR}-${IMODEL}/line-numbers-${TARGET_WORD}.npy
}

main() {
  # # Skip if the model file already exists
  # MODEL_FILE_NAME=${MODEL_DIR}/model-${TARGET_WORD}.ptch

  # if [ -f "$MODEL_FILE_NAME" ]; then
  #   echo "Model file already exists. Skipping training."
  #   exit 0
  # fi


  # sample_data
  # augment_data
  encode_sentences
  train_encoder
  combine_models
  #  # Validate the model
  #  download_validation_data
   embed_sentences
   visualize_embeddings
  # cleaenup
}

main "$@"