# Train one 15-language many-to-many Marian model from Tatoeba using two
# reliable pivots:
#   en -> all, all -> en, tr -> all, all -> tr
#
# This produces 54 unique training directions. Every direction gets 500
# held-out Tatoeba examples. BOUQuET contributes 500 examples to each selected
# direction not involving Standard Arabic, and each corpus/direction is logged
# independently during validation.
#
# No pretrained model is configured, so this starts from scratch.
#
# Usage:
#   make -f configs/en-multi15/tatooeba.en-tr-pivots.bouquet.transformer.mk \
#        -f Makefile data
#   make -f configs/en-multi15/tatooeba.en-tr-pivots.bouquet.transformer.mk \
#        -f Makefile train

LANGS = en tr kmr de es fr el bg ru ka hy fa ar ckb ur
SRCLANGS = ${LANGS}
TRGLANGS = ${LANGS}
LANGPAIRSTR = tatooeba-en-tr-pivots

MIX_DATASET = tatooeba-en-tr-pivots
MIX_USE_HIGH_QUALITY = 0
MIX_BOUQUET_DEV_ROWS = 500

# First pass: en/tr -> every target (28 directions).
MIX_PRIMARY_SRCLANGS = en tr
MIX_PRIMARY_TRGLANGS = ${LANGS}
MIX_PRIMARY_MODE = all-to-all
MIX_PRIMARY_SOURCE_LANG =
MIX_PRIMARY_PAIRS = ${foreach s,${MIX_PRIMARY_SRCLANGS},${foreach t,${MIX_PRIMARY_TRGLANGS},${if ${filter ${s},${t}},,${s}-${t}}}}

# Second pass: every remaining source -> en/tr (26 directions). Excluding en
# and tr here prevents duplicate en->tr and tr->en examples.
MIX_SECONDARY_SRCLANGS = ${filter-out en tr,${LANGS}}
MIX_SECONDARY_TRGLANGS = en tr
MIX_SECONDARY_MODE = all-to-all
MIX_SECONDARY_PAIRS = ${foreach s,${MIX_SECONDARY_SRCLANGS},${foreach t,${MIX_SECONDARY_TRGLANGS},${s}-${t}}}

MIX_PROJECTED_PAIRS = ${MIX_PRIMARY_PAIRS} ${MIX_SECONDARY_PAIRS}
MIX_EVAL_PAIRS = ${MIX_PROJECTED_PAIRS}

HF_CSV_NWAY_MODE = all-to-all
HF_CSV_NWAY_SOURCE_LANG =

BOUQUET_SRCLANGS = en tr kmr de es fr el bg ru ka hy fa ckb ur
BOUQUET_TRGLANGS = ${BOUQUET_SRCLANGS}

DEV_SRCLANGS = ${SRCLANGS}
DEV_TRGLANGS = ${TRGLANGS}

include configs/en-multi15/_tatooeba-hq.common.mk
