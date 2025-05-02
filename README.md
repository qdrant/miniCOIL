
# miniCOIL

MiniCOIL is a contextualized per-word embedding model.
MiniCOIL generates small-size embeddings for each word in a sentence, but the embeddings can only be compared within 
embeddings of the same words in different sentences(context). 
This restriction allows to generate an extremely small embeddings (8d or even 4d) while still preserving the context of the word.

## Usage

MiniCOIL embeddings might be useful in information retrieval tasks, where we need to resolve meaning of the word in the context of the sentence.
For example, many words have different meanings depending on the context, such as "bank" (river bank or financial institution).

MiniCOIL allows to encode precise meaning of the word, but unlike traditional word embeddings it won't dilute exact match with other words in the vocabulary.

MiniCOIL is not trained in end-to-end fashion, which means that it can't assign relative importance to the words in the sentence.
However, it can be combined with BM25-like scoring formula and used in search engines.

## Architecture

MiniCOIL is designed to be compatible with foundational transformer models, such as SentenceTransformers.
There are two main reasons for this:

- We don't want to spend enormous resources on training MiniCOIL.
- We want to be able to combine MiniCOIL embeddings inference with dense embedding inference in a single step.

Technically, MiniCOIL is a simple array of linear layers (one for each word in vocabulary) that are trained 
to compress the word embeddings into a small size. That makes MiniCOIL a paper-thin layer on top of the transformer model.

### Training process

MiniCOIL is trained by the principle skip-gram models, adapted to the transformer model: we want to predict the context by the word.
In case of the transformer models, we predict sentence embeddings by the word embeddings.

Naturally, this process can be separated into two steps: encoding and deciding (similar to autoencoders), where in the middle we have a small-size embeddings.

Since we want to make MiniCOIL compatible with many transformer models, we can replace the decoder step with compressed embeddings of some larger model,
so for each input model we can train the encoder independently.

So the process of training is as follows:

1. Download dataset (we use openwebtext)
1. Convert dataset into readable format with `mini_coil.data_pipeline.convert_openwebtext`
1. Split data into sentences with `mini_coil.data_pipeline.split_sentences`
1. Encode sentences with transformer model, save embeddings to disk (about 350M embeddings for openwebtext) with `mini_coil.data_pipeline.encode_targets`
1. Upload encoded sentences to Qdrant, so we can sample sentences with specified words with `mini_coil.data_pipeline.upload_to_qdrant`
1. For triplet-based training - follow [train-triplets.sh](./train-triplets.sh). It will for each word:
   1. Denerate Distance Matrix based on large embeddings
   2. Augment sentences
   3. Encode sentences with small model
   4. Train per-word encoder
1. Merge encoders for each word into a single model `mini_coil.data_pipeline.combine_models`
1. Make visualizations
1. Benchmark
1. Quantize (optional)
1. Convert to ONNX
