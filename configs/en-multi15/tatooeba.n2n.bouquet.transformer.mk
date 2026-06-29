# Train a 15-language many-to-many Marian model from:
#   1x TartarusXXX/tatooeba-synthetic
#
# Training covers every non-identical language direction. Validation is limited
# to en->all, all->en, tr->all, and all->tr. Every selected direction gets 500
# held-out Tatoeba examples; selected BOUQuET directions not involving Standard
# Arabic also get 500 examples. The two corpora are logged independently.
#
# No N-to-N pretrained model is configured, so this starts from scratch.
#
# Usage:
#   make -f configs/en-multi15/tatooeba.n2n.bouquet.transformer.mk \
#        -f Makefile data
#   make -f configs/en-multi15/tatooeba.n2n.bouquet.transformer.mk \
#        -f Makefile train

LANGS = en tr kmr de es fr el bg ru ka hy fa ar ckb ur
SRCLANGS = ${LANGS}
TRGLANGS = ${LANGS}
LANGPAIRSTR = tatooeba-n2n

# Match hf_csv2moses.py all-to-all ordering exactly: source-major, then target.
MIX_DATASET = tatooeba-n2n
MIX_PROJECTED_PAIRS = ${foreach s,${SRCLANGS},${foreach t,${TRGLANGS},${if ${filter ${s},${t}},,${s}-${t}}}}
MIX_EVAL_PAIRS = ${sort \
	${foreach t,${filter-out en,${TRGLANGS}},en-${t}} \
	${foreach s,${filter-out en,${SRCLANGS}},${s}-en} \
	${foreach t,${filter-out tr,${TRGLANGS}},tr-${t}} \
	${foreach s,${filter-out tr,${SRCLANGS}},${s}-tr}}
MIX_USE_HIGH_QUALITY = 0
MIX_BOUQUET_DEV_ROWS = 500

HF_CSV_NWAY_MODE = all-to-all
HF_CSV_NWAY_SOURCE_LANG =

BOUQUET_SRCLANGS = en tr kmr de es fr el bg ru ka hy fa ckb ur
BOUQUET_TRGLANGS = ${BOUQUET_SRCLANGS}

DEV_SRCLANGS = ${SRCLANGS}
DEV_TRGLANGS = ${TRGLANGS}

include configs/en-multi15/_tatooeba-hq.common.mk
