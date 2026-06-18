# Train an English-to-many multilingual model from the private Hugging Face
# 16-language CSV dataset and evaluate on English-source BOUQuET dev data.

export LANGS SRCLANGS TRGLANGS LANGPAIRSTR
export HF_CSV_REPO HF_CSV_FILE HF_CSV_CORPUS HF_CSV_REVISION
export HF_CSV_NWAY_TRAIN HF_CSV_NWAY_SOURCE_LANG HF_CSV_NWAY_SHUFFLE_BUFFER HF_CSV_NWAY_SHUFFLE_SEED
export HF_CSV_SKIP_PAIR_DATA
export DATASET TRAINSET
export EVAL_LANGS DEVSET DEVSET_NAME DEV_SRCLANGS DEV_TRGLANGS DEVSET_BY_TRGLANG KEEP_FULL_DEVSET
export TESTSET TESTSET_NAME TEST_SRCLANGS TEST_TRGLANGS TESTSET_BY_TRGLANG KEEP_FULL_TESTSET
export USE_BOUQUET_DATA BOUQUET_LANGS BOUQUET_SRCLANGS BOUQUET_TRGLANGS
export TARGET_LABEL_BY_TRGLANG
export MODELTYPE PRE SUBWORDS CLEAN_TRAINDATA_TYPE CLEAN_DEVDATA_TYPE CLEAN_TESTDATA_TYPE
export SKIP_SAME_LANG SHUFFLE_DATA SHUFFLE_MULTILINGUAL_DATA DATA_IS_SHUFFLED
export SPM_INPUT_SIZE USE_REST_DEVDATA FIT_DEVDATA_SIZE
export DEVSIZE TESTSIZE DEVMINSIZE SUBWORD_VOCAB_SIZE MAX_OVER_SAMPLING
export MARIAN_VALID_METRICS MARIAN_VALID_FREQ MARIAN_MINI_BATCH MARIAN_MAXI_BATCH MARIAN_VALID_MINI_BATCH MARIAN_EXTRA MARIAN_WORKSPACE
export GPUJOB_HPC_MEM GPUJOB_SUBMIT

LANGS       = en tr kmr de es fr el bg ru ka hy fa ar ckb ur
SRCLANGS    = en
TRGLANGS    = ${LANGS}
LANGPAIRSTR = synthetic-en2n-bouquet

HF_CSV_REPO     = TartarusXXX/synthetic-parallel-16lang-1-4m-gemini-3-1-flash-lite
HF_CSV_FILE     = synthetic_parallel_16lang_1_4m_gemini_3_1_flash_lite.csv
HF_CSV_CORPUS   = synthetic_parallel_16lang_1_4m_gemini_3_1_flash_lite
HF_CSV_REVISION = main

DATASET  = ${HF_CSV_CORPUS}
TRAINSET = ${HF_CSV_CORPUS}

# The cached train stream is already projected from English to every target
# language with target labels on the source side.
HF_CSV_NWAY_TRAIN = 1
HF_CSV_NWAY_SOURCE_LANG = en
HF_CSV_NWAY_SHUFFLE_BUFFER = 200000
HF_CSV_NWAY_SHUFFLE_SEED = 1
HF_CSV_SKIP_PAIR_DATA = 1

EVAL_LANGS = tr kmr de es fr el bg ru ka hy fa ckb ur apc arz
USE_BOUQUET_DATA = 1
BOUQUET_LANGS = tur_Latn:tr eng_Latn:en deu_Latn:de spa_Latn:es fra_Latn:fr ell_Grek:el bul_Cyrl:bg rus_Cyrl:ru kat_Geor:ka hye_Armn:hy pes_Arab:fa ckb_Arab:ckb urd_Arab:ur kmr_Latn:kmr apc_Arab:apc arz_Arab:arz
BOUQUET_SRCLANGS = en
BOUQUET_TRGLANGS = ${EVAL_LANGS}
TARGET_LABEL_BY_TRGLANG = apc:ar arz:ar

DEVSET   = bouquet-dev-en2n
DEVSET_NAME = bouquet-dev-en2n
DEV_SRCLANGS = en
DEV_TRGLANGS = ${EVAL_LANGS}
DEVSET_BY_TRGLANG = tr:bouquet-dev-en2n kmr:bouquet-dev-en2n de:bouquet-dev-en2n es:bouquet-dev-en2n fr:bouquet-dev-en2n el:bouquet-dev-en2n bg:bouquet-dev-en2n ru:bouquet-dev-en2n ka:bouquet-dev-en2n hy:bouquet-dev-en2n fa:bouquet-dev-en2n ckb:bouquet-dev-en2n ur:bouquet-dev-en2n apc:bouquet-dev-en2n arz:bouquet-dev-en2n
KEEP_FULL_DEVSET = 1

TESTSET  = bouquet-dev-en2n
TESTSET_NAME = bouquet-dev-en2n
TEST_SRCLANGS = en
TEST_TRGLANGS = ${EVAL_LANGS}
TESTSET_BY_TRGLANG = ${DEVSET_BY_TRGLANG}
KEEP_FULL_TESTSET = 1

MODELTYPE = transformer
PRE       = simple
SUBWORDS  = spm
CLEAN_TRAINDATA_TYPE = clean
CLEAN_DEVDATA_TYPE = clean
CLEAN_TESTDATA_TYPE = clean

SKIP_SAME_LANG = 1
SHUFFLE_DATA = 1
SHUFFLE_MULTILINGUAL_DATA = 0
DATA_IS_SHUFFLED = 1
SPM_INPUT_SIZE = 10000000
USE_REST_DEVDATA = 1
FIT_DEVDATA_SIZE =

DEVSIZE = 2500
TESTSIZE = 2500
DEVMINSIZE = 250

SUBWORD_VOCAB_SIZE = 32000
MAX_OVER_SAMPLING = 50

MARIAN_VALID_METRICS = bleu
MARIAN_VALID_FREQ = 26500u
MARIAN_MINI_BATCH = 2048
MARIAN_MAXI_BATCH = 4000
MARIAN_VALID_MINI_BATCH = 32
MARIAN_EXTRA += --mini-batch ${MARIAN_MINI_BATCH} --maxi-batch ${MARIAN_MAXI_BATCH}
MARIAN_WORKSPACE = 30000

GPUJOB_HPC_MEM = 16g
GPUJOB_SUBMIT =
