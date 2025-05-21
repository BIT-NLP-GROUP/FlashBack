# RAG-is-all-you-need

# why RAG

1.Explicitly memorizing the training data helps generation

2.LMs can scale to larger text collections without the added cost of training,
by simply adding the data to the index

3.A single LM can adapt to multiple domains without the in-domain training,
by adding domain-specific data to the inde

# Existing work

* KNN-LM——Token-level and Interpolation-based model
* RETRO——Chunk-level, Frozen-Retriever, huge index model
* REALM——Document-level and Joint-Training model

# Open-domain QA Dataset:

Natural Questions, TriviaQA (ICL RAG 测了这两个)，WebQuestions, CuratedTrec

Baselines:

* ORQA (Lee et al. 2019) – 330M paras

  * Equivalent to REALM without joint training
* T5-base (220M), L (770M), XL (11B) (Raffel et al. 2019)

# Experiments

### Run-time Improvement

use `test_speed.sh`.

### Language Modeling

use `train.sh` and `eval.sh`.

### Multi-K

use `test_multik.sh`.
