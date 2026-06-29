# Fine-tune the existing many-to-English Marian model on:
#   1x TartarusXXX/tatooeba-synthetic
#   1x TartarusXXX/agentlans-high-quality-english-sentences-synthetic
#
# BOUQuET and 500 held-out Tatoeba rows per source are scored independently.
# Standard Arabic has no BOUQuET split, so ar uses only the Tatoeba holdout.
#
# Usage:
#   make -f configs/en-multi15/tatooeba-hq.all2en.bouquet.transformer.mk \
#        -f Makefile data
#   make -f configs/en-multi15/tatooeba-hq.all2en.bouquet.transformer.mk \
#        -f Makefile train

LANGS = en tr kmr de es fr el bg ru ka hy fa ar ckb ur
SRCLANGS = tr kmr de es fr el bg ru ka hy fa ar ckb ur
TRGLANGS = en
LANGPAIRSTR = tatooeba-hq-all2en

MIX_EVAL_PAIRS = ${foreach s,${SRCLANGS},${s}-en}

# all-to-all with one target column projects every selected source to English.
HF_CSV_NWAY_MODE = all-to-all
HF_CSV_NWAY_SOURCE_LANG =

HF_MARIAN_MODEL_REPO = TartarusXXX/synthetic-n2en-marian

BOUQUET_SRCLANGS = tr kmr de es fr el bg ru ka hy fa ckb ur
BOUQUET_TRGLANGS = en

DEV_SRCLANGS = ${SRCLANGS}
DEV_TRGLANGS = en

include configs/en-multi15/_tatooeba-hq.common.mk
