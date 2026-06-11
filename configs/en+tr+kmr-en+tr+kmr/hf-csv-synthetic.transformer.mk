# Sample config for training an en/tr/kmr multilingual model from a
# private Hugging Face CSV dataset with language-code headers.
#
# Usage:
#   make -f configs/en+tr+kmr-en+tr+kmr/hf-csv-synthetic.transformer.mk -f Makefile hf-csv-multilingual
#
# The downloader reads HF_TOKEN from the environment or .env.

LANGS       = en tr kmr
SRCLANGS    = ${LANGS}
TRGLANGS    = ${LANGS}
LANGPAIRSTR = en+tr+kmr-en+tr+kmr

HF_CSV_REPO     = TartarusXXX/synthetic-en-tr-kmr-1-4m-gemini-3-1-flash-lite
HF_CSV_FILE     = synthetic_en_tr_kmr_1_4m_gemini_3_1_flash_lite.csv
HF_CSV_CORPUS   = synthetic_en_tr_kmr_1_4m_gemini_3_1_flash_lite
HF_CSV_REVISION = main

DATASET  = ${HF_CSV_CORPUS}
TRAINSET =
DEVSET   = ${HF_CSV_CORPUS}
TESTSET  = ${HF_CSV_CORPUS}

# Optional: use external per-direction or per-target-language test sets
# for evaluation, for example Bouquet. Pair mappings override target
# language mappings when both are set.
#
# TESTSET_NAME = bouquet
# TESTSET_BY_LANGPAIR = en-tr:bouquet-en-tr tr-en:bouquet-tr-en \
#                       en-kmr:bouquet-en-kmr kmr-en:bouquet-kmr-en \
#                       tr-kmr:bouquet-tr-kmr kmr-tr:bouquet-kmr-tr
#
# TESTSET_BY_TRGLANG = en:bouquet-en tr:bouquet-tr kmr:bouquet-kmr

MODELTYPE = transformer
PRE       = simple
SUBWORDS  = spm
CLEAN_TRAINDATA_TYPE = clean
CLEAN_DEVDATA_TYPE = clean
CLEAN_TESTDATA_TYPE = clean

SKIP_SAME_LANG = 1
SHUFFLE_DATA = 1
SHUFFLE_MULTILINGUAL_DATA = 1
USE_REST_DEVDATA = 1
FIT_DEVDATA_SIZE =

DEVSIZE = 2500
TESTSIZE = 2500
DEVMINSIZE = 250

SUBWORD_VOCAB_SIZE = 32000
MAX_OVER_SAMPLING = 50

MARIAN_VALID_FREQ = 5000u
MARIAN_SAVE_FREQ = 5000u
MARIAN_DISP_FREQ = 1000u
MARIAN_VALID_METRICS = bleu chrf perplexity
MARIAN_WORKSPACE = 10000
MARIAN_VALID_MINI_BATCH = 8

GPUJOB_HPC_MEM = 8g
GPUJOB_SUBMIT =
