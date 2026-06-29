# Fine-tune the existing English-to-many Marian model on:
#   1x TartarusXXX/tatooeba-synthetic
#   1x TartarusXXX/agentlans-high-quality-english-sentences-synthetic
#
# BOUQuET and 500 held-out Tatoeba rows per target are scored independently.
# Standard Arabic has no BOUQuET split, so ar uses only the Tatoeba holdout.
#
# Usage:
#   make -f configs/en-multi15/tatooeba-hq.en2all.bouquet.transformer.mk \
#        -f Makefile data
#   make -f configs/en-multi15/tatooeba-hq.en2all.bouquet.transformer.mk \
#        -f Makefile train

LANGS = en tr kmr de es fr el bg ru ka hy fa ar ckb ur
SRCLANGS = en
TRGLANGS = tr kmr de es fr el bg ru ka hy fa ar ckb ur
LANGPAIRSTR = tatooeba-hq-en2all

MIX_EVAL_PAIRS = ${foreach t,${TRGLANGS},en-${t}}

HF_CSV_NWAY_MODE = one-to-many
HF_CSV_NWAY_SOURCE_LANG = en

HF_MARIAN_MODEL_REPO = TartarusXXX/synthetic-en2n-marian

BOUQUET_SRCLANGS = en
BOUQUET_TRGLANGS = tr kmr de es fr el bg ru ka hy fa ckb ur

DEV_SRCLANGS = en
DEV_TRGLANGS = ${TRGLANGS}

include configs/en-multi15/_tatooeba-hq.common.mk
