# Train a multilingual model from the private Hugging Face 16-language CSV
# dataset and evaluate on dedicated BOUQuET dev testsets.
#
# Usage:
#   make -f configs/en+tr+kmr-en+tr+kmr/hf-csv-synthetic.bouquet.transformer-big.mk -f Makefile MARIAN_GPUS=0 hf-csv-multilingual

LANGS       = en tr kmr de es fr el bg ru ka hy fa ar ckb ur
SRCLANGS    = ${LANGS}
TRGLANGS    = ${LANGS}
LANGPAIRSTR = synthetic-15lang-bouquet-big

HF_CSV_REPO     = TartarusXXX/synthetic-parallel-16lang-1-4m-gemini-3-1-flash-lite
HF_CSV_FILE     = synthetic_parallel_16lang_1_4m_gemini_3_1_flash_lite.csv
HF_CSV_CORPUS   = synthetic_parallel_16lang_1_4m_gemini_3_1_flash_lite
HF_CSV_REVISION = main

DATASET  = ${HF_CSV_CORPUS}
TRAINSET = ${HF_CSV_CORPUS}

# This CSV is N-way parallel. Build the local Marian train streams directly
# from each CSV row instead of expanding the row into every language pair.
HF_CSV_NWAY_TRAIN = 1
HF_CSV_NWAY_SOURCE_LANG = en
HF_CSV_NWAY_SHUFFLE_BUFFER = 200000
HF_CSV_NWAY_SHUFFLE_SEED = 1

# Use BOUQuET as the Marian validation set during training, with a dedicated
# corpus per target language. BOUQuET does not provide standard Arabic in this
# split, so ar stays in training and Levantine/Egyptian Arabic are used for
# Arabic evaluation.
EVAL_LANGS = en tr kmr de es fr el bg ru ka hy fa ckb ur apc arz
DEVSET   = bouquet-dev
DEVSET_NAME = bouquet-dev
DEV_SRCLANGS = ${EVAL_LANGS}
DEV_TRGLANGS = ${EVAL_LANGS}
DEVSET_BY_TRGLANG = en:bouquet-dev-en tr:bouquet-dev-tr kmr:bouquet-dev-kmr de:bouquet-dev-de es:bouquet-dev-es fr:bouquet-dev-fr el:bouquet-dev-el bg:bouquet-dev-bg ru:bouquet-dev-ru ka:bouquet-dev-ka hy:bouquet-dev-hy fa:bouquet-dev-fa ckb:bouquet-dev-ckb ur:bouquet-dev-ur apc:bouquet-dev-apc arz:bouquet-dev-arz
KEEP_FULL_DEVSET = 1

# Keep the same BOUQuET mapping for final test evaluation.
TESTSET  = bouquet-dev
TESTSET_NAME = bouquet-dev
TEST_SRCLANGS = ${EVAL_LANGS}
TEST_TRGLANGS = ${EVAL_LANGS}
TESTSET_BY_TRGLANG = en:bouquet-dev-en tr:bouquet-dev-tr kmr:bouquet-dev-kmr de:bouquet-dev-de es:bouquet-dev-es fr:bouquet-dev-fr el:bouquet-dev-el bg:bouquet-dev-bg ru:bouquet-dev-ru ka:bouquet-dev-ka hy:bouquet-dev-hy fa:bouquet-dev-fa ckb:bouquet-dev-ckb ur:bouquet-dev-ur apc:bouquet-dev-apc arz:bouquet-dev-arz
KEEP_FULL_TESTSET = 1

MODELTYPE = transformer-big
PRE       = simple
SUBWORDS  = spm
CLEAN_TRAINDATA_TYPE = clean
CLEAN_DEVDATA_TYPE = clean
CLEAN_TESTDATA_TYPE = clean

SKIP_SAME_LANG = 1
SHUFFLE_DATA = 1
# Pair data is already shuffled; avoid the redundant global multilingual sort
# because the 15-language aggregate is too large for /tmp scratch.
SHUFFLE_MULTILINGUAL_DATA = 0
# The expanded N-way corpus is already shuffled enough for subword training.
# Avoid an extra random sort in SentencePiece preprocessing and keep the
# temporary SPM text sample bounded.
DATA_IS_SHUFFLED = 1
SPM_INPUT_SIZE = 10000000
USE_REST_DEVDATA = 1
FIT_DEVDATA_SIZE =

DEVSIZE = 2500
TESTSIZE = 2500
DEVMINSIZE = 250

SUBWORD_VOCAB_SIZE = 32000
MAX_OVER_SAMPLING = 50

MARIAN_VALID_METRICS = bleu chrf perplexity
MARIAN_MINI_BATCH = 512
MARIAN_MAXI_BATCH = 1000
MARIAN_VALID_MINI_BATCH = 16
MARIAN_WORKSPACE = 20000

GPUJOB_HPC_MEM = 16g
GPUJOB_SUBMIT =
