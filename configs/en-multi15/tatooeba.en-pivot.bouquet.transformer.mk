# Train one 15-language Marian model from Tatoeba using English as the only
# pivot:
#   en -> all, all -> en
#
# This produces 28 unique training directions. Every direction gets 500
# held-out Tatoeba examples. BOUQuET contributes 500 examples to each selected
# direction not involving Standard Arabic, and each corpus/direction is logged
# independently during validation.
#
# No pretrained model is configured, so this starts from scratch.
#
# Usage:
#   make -f configs/en-multi15/tatooeba.en-pivot.bouquet.transformer.mk \
#        -f Makefile data
#   make -f configs/en-multi15/tatooeba.en-pivot.bouquet.transformer.mk \
#        -f Makefile train

LANGS = en tr kmr de es fr el bg ru ka hy fa ar ckb ur
SRCLANGS = ${LANGS}
TRGLANGS = ${LANGS}
LANGPAIRSTR = tatooeba-en-pivot

MIX_DATASET = tatooeba-en-pivot
MIX_USE_HIGH_QUALITY = 0
MIX_BOUQUET_DEV_ROWS = 500

# First pass: English -> every other language (14 directions).
MIX_PRIMARY_SRCLANGS = en
MIX_PRIMARY_TRGLANGS = ${LANGS}
MIX_PRIMARY_MODE = one-to-many
MIX_PRIMARY_SOURCE_LANG = en
MIX_PRIMARY_PAIRS = ${foreach t,${filter-out en,${MIX_PRIMARY_TRGLANGS}},en-${t}}

# Second pass: every other language -> English (14 directions).
MIX_SECONDARY_SRCLANGS = ${filter-out en,${LANGS}}
MIX_SECONDARY_TRGLANGS = en
MIX_SECONDARY_MODE = all-to-all
MIX_SECONDARY_PAIRS = ${foreach s,${MIX_SECONDARY_SRCLANGS},${s}-en}

MIX_PROJECTED_PAIRS = ${MIX_PRIMARY_PAIRS} ${MIX_SECONDARY_PAIRS}
MIX_EVAL_PAIRS = ${MIX_PROJECTED_PAIRS}

HF_CSV_NWAY_MODE = all-to-all
HF_CSV_NWAY_SOURCE_LANG =

BOUQUET_SRCLANGS = en tr kmr de es fr el bg ru ka hy fa ckb ur
BOUQUET_TRGLANGS = ${BOUQUET_SRCLANGS}

DEV_SRCLANGS = ${SRCLANGS}
DEV_TRGLANGS = ${TRGLANGS}

include configs/en-multi15/_tatooeba-hq.common.mk
