# Shared settings and data preparation for synthetic N-way configs. Include
# this after setting the languages, projection mode, and MIX_EVAL_PAIRS.

export LANGS SRCLANGS TRGLANGS LANGPAIRSTR
export DATASET TRAINSET
export HF_CSV_REPO HF_CSV_FILE HF_CSV_CORPUS HF_CSV_REVISION
export HF_CSV_NWAY_TRAIN HF_CSV_NWAY_MODE HF_CSV_NWAY_SOURCE_LANG
export HF_CSV_NWAY_SHUFFLE_BUFFER HF_CSV_SKIP_PAIR_DATA
export HF_MARIAN_MODEL_REPO HF_MARIAN_MODEL_REVISION
export USE_BOUQUET_DATA BOUQUET_LANGS BOUQUET_SRCLANGS BOUQUET_TRGLANGS
export BOUQUET_CORPUS_PREFIX BOUQUET_CORPUS_MODES
export DEVSET DEVSET_NAME DEV_SRCLANGS DEV_TRGLANGS KEEP_FULL_DEVSET
export TESTSET TESTSET_NAME TEST_SRCLANGS TEST_TRGLANGS KEEP_FULL_TESTSET
export MARIAN_NAMED_VALIDATION NAMED_VALIDATION_AUTO_MANIFEST
export MODELTYPE PRE SUBWORDS
export CLEAN_TRAINDATA_TYPE CLEAN_DEVDATA_TYPE CLEAN_TESTDATA_TYPE
export SKIP_SAME_LANG SHUFFLE_DATA SHUFFLE_MULTILINGUAL_DATA
export USE_REST_DEVDATA FIT_DEVDATA_SIZE
export SUBWORD_VOCAB_SIZE
export MARIAN_VALID_METRICS MARIAN_VALID_FREQ MARIAN_SAVE_FREQ
export MARIAN_DISP_FREQ MARIAN_EARLY_STOPPING MARIAN_VALID_MINI_BATCH
export MARIAN_MINI_BATCH MARIAN_MAXI_BATCH MARIAN_EXTRA
export MARIAN_WORKSPACE MARIAN_SHUFFLE
export GPUJOB_HPC_MEM GPUJOB_SUBMIT

TATOOEBA_HF_REPO = TartarusXXX/tatooeba-synthetic
TATOOEBA_HF_FILE = data/train.csv
TATOOEBA_HF_CORPUS = tatooeba-synthetic

HIGH_QUALITY_HF_REPO = TartarusXXX/agentlans-high-quality-english-sentences-synthetic
HIGH_QUALITY_HF_FILE = data/train.csv
HIGH_QUALITY_HF_CORPUS = agentlans-high-quality-english-sentences-synthetic

MIX_DATASET ?= tatooeba-hq-mix

# Keep the standard HF N-way integration selected for downstream data rules.
# The custom prerequisite below creates the mixed local train stream first, so
# the single-repository fallback is not used.
HF_CSV_REPO = ${TATOOEBA_HF_REPO}
HF_CSV_FILE = ${TATOOEBA_HF_FILE}
HF_CSV_CORPUS = ${MIX_DATASET}
HF_CSV_REVISION = main
HF_CSV_NWAY_TRAIN = 1
HF_CSV_SKIP_PAIR_DATA = 1
HF_CSV_NWAY_SHUFFLE_BUFFER =

DATASET = ${MIX_DATASET}
TRAINSET = ${DATASET}

HF_MARIAN_MODEL_REVISION = main

# BOUQuET is an explicit prerequisite of hf-mixed-nway-data. Keep the generic
# rawdata target from invoking the converter a second time.
USE_BOUQUET_DATA = 0
BOUQUET_LANGS = tur_Latn:tr eng_Latn:en deu_Latn:de spa_Latn:es \
		fra_Latn:fr ell_Grek:el bul_Cyrl:bg rus_Cyrl:ru \
		kat_Geor:ka hye_Armn:hy pes_Arab:fa ckb_Arab:ckb \
		urd_Arab:ur kmr_Latn:kmr
BOUQUET_CORPUS_PREFIX = bouquet-dev
BOUQUET_CORPUS_MODES = target

DEVSET = bouquet+tatooeba-dev
DEVSET_NAME = bouquet+tatooeba-dev
KEEP_FULL_DEVSET = 1

# Final test evaluation reuses the same held-out streams. Named scores used
# during training remain split by corpus and language in the sidecar TSV.
TESTSET = ${DEVSET}
TESTSET_NAME = ${DEVSET_NAME}
TEST_SRCLANGS = ${DEV_SRCLANGS}
TEST_TRGLANGS = ${DEV_TRGLANGS}
KEEP_FULL_TESTSET = 1

MARIAN_NAMED_VALIDATION = 1
NAMED_VALIDATION_AUTO_MANIFEST = 0

MODELTYPE = transformer
PRE = simple
SUBWORDS = spm
CLEAN_TRAINDATA_TYPE = clean
CLEAN_DEVDATA_TYPE = clean
CLEAN_TESTDATA_TYPE = clean

SKIP_SAME_LANG = 1
SHUFFLE_DATA = 0
SHUFFLE_MULTILINGUAL_DATA = 0
USE_REST_DEVDATA = 0
FIT_DEVDATA_SIZE =

SUBWORD_VOCAB_SIZE = 32000

MARIAN_VALID_METRICS = bleu
MARIAN_VALID_FREQ = 26500u
MARIAN_SAVE_FREQ = 26500u
MARIAN_DISP_FREQ = 1000u
MARIAN_EARLY_STOPPING = 10
MARIAN_VALID_MINI_BATCH = 32
MARIAN_MINI_BATCH = 2048
MARIAN_MAXI_BATCH = 4000
MARIAN_WORKSPACE = 30000
# The two projected corpora are concatenated. Let Marian reshuffle examples
# between epochs instead of globally sorting tens of millions of text lines.
MARIAN_SHUFFLE = data

GPUJOB_HPC_MEM = 16g
GPUJOB_SUBMIT =

MIX_TATOOEBA_DEV_ROWS ?= 500
MIX_BOUQUET_DEV_ROWS ?=
MIX_USE_HIGH_QUALITY ?= 1
MIX_PROJECTED_PAIRS ?= ${MIX_EVAL_PAIRS}
MIX_PRIMARY_SRCLANGS ?= ${SRCLANGS}
MIX_PRIMARY_TRGLANGS ?= ${TRGLANGS}
MIX_PRIMARY_MODE ?= ${HF_CSV_NWAY_MODE}
MIX_PRIMARY_SOURCE_LANG ?= ${HF_CSV_NWAY_SOURCE_LANG}
MIX_PRIMARY_PAIRS ?= ${MIX_PROJECTED_PAIRS}
MIX_SECONDARY_SRCLANGS ?=
MIX_SECONDARY_TRGLANGS ?=
MIX_SECONDARY_MODE ?= all-to-all
MIX_SECONDARY_SOURCE_LANG ?=
MIX_SECONDARY_PAIRS ?=
MIX_HQ_SHUFFLE_BUFFER ?= 200000
MIX_HF_CACHE_DIR ?= ${WORKHOME}/data/hf
MIX_HF_CONVERTER ?= ${REPOHOME}scripts/hf_csv2moses.py
MIX_BOUQUET_TARGET ?= bouquet-data

.PHONY: hf-mixed-nway-data
data train: hf-mixed-nway-data

# Project each selected CSV exactly once. Tatoeba is kept row-major long enough
# to hold out the first 500 valid rows for every direction. All remaining
# Tatoeba data is used for training; high-quality data is optional.
hf-mixed-nway-data: ${MIX_BOUQUET_TARGET}
	@set -euo pipefail; \
	mkdir -p "${dir ${LOCAL_TRAIN_SRC}}" "${dir ${DEV_SRC}}" "${dir ${TEST_SRC}}"; \
	if { test -s "${LOCAL_TRAIN_SRC}" && test -s "${LOCAL_TRAIN_TRG}"; } || \
	   { test -s "$(strip ${TRAINDATA_SRC})" && test -s "$(strip ${TRAINDATA_TRG})"; }; then \
	  if test -s "${DEV_SRC}" && test -s "${DEV_TRG}" && \
	     test -s "${TEST_SRC}" && test -s "${TEST_TRG}" && \
	     test -s "${NAMED_VALIDATION_MANIFEST}"; then \
	    echo "mixed HF train and validation data already prepared"; \
	    exit 0; \
	  fi; \
	fi; \
	base="${TMPWORKDIR}/${LANGPAIRSTR}/hf-mix"; \
	tato_src="$$base.tatooeba.all.src"; \
	tato_trg="$$base.tatooeba.all.trg"; \
	tato_meta="$$base.tatooeba.metadata.json"; \
	tato_secondary_src="$$base.tatooeba.secondary.src"; \
	tato_secondary_trg="$$base.tatooeba.secondary.trg"; \
	tato_secondary_meta="$$base.tatooeba.secondary.metadata.json"; \
	tato_train_src="$$base.tatooeba.train.src"; \
	tato_train_trg="$$base.tatooeba.train.trg"; \
	tato_dev_base="$$base.tatooeba.dev"; \
	hq_src="$$base.high-quality.src"; \
	hq_trg="$$base.high-quality.trg"; \
	hq_meta="$$base.high-quality.metadata.json"; \
	rm -f "$$tato_src" "$$tato_trg" "$$tato_meta" \
	      "$$tato_secondary_src" "$$tato_secondary_trg" "$$tato_secondary_meta" \
	      "$$tato_train_src" "$$tato_train_trg" \
	      "$$tato_dev_base".*.src "$$tato_dev_base".*.trg \
	      "$$hq_src" "$$hq_trg" "$$hq_meta" \
	      "${LOCAL_TRAIN_SRC}" "${LOCAL_TRAIN_TRG}" \
	      "${DEV_SRC}" "${DEV_TRG}" "${TEST_SRC}" "${TEST_TRG}" \
	      "${NAMED_VALIDATION_MANIFEST}"; \
	"${MIX_HF_CONVERTER}" \
	  --repo-id "${TATOOEBA_HF_REPO}" \
	  --filename "${TATOOEBA_HF_FILE}" \
	  --revision main \
	  --cache-dir "${MIX_HF_CACHE_DIR}" \
	  --src-langs ${MIX_PRIMARY_SRCLANGS} \
	  --trg-langs ${MIX_PRIMARY_TRGLANGS} \
	  --corpus "${TATOOEBA_HF_CORPUS}" \
	  --nway-train-src "$$tato_src" \
	  --nway-train-trg "$$tato_trg" \
	  --nway-mode "${MIX_PRIMARY_MODE}" \
	  --nway-metadata "$$tato_meta" \
	  $(if ${MIX_PRIMARY_SOURCE_LANG},--nway-source-lang "${MIX_PRIMARY_SOURCE_LANG}") \
	  --keep-empty \
	  --skip-same-lang \
	  --overwrite; \
	: > "$$tato_train_src"; \
	: > "$$tato_train_trg"; \
	split_tatoeba_projection() { \
	  projection_src="$$1"; \
	  projection_trg="$$2"; \
	  projected_pairs="$$3"; \
	  paste "$$projection_src" "$$projection_trg" | \
	    awk -F '\t' -v projected="$$projected_pairs" \
	      -v evaluated="${MIX_EVAL_PAIRS}" -v keep="${MIX_TATOOEBA_DEV_ROWS}" \
	      -v dev="$$tato_dev_base" \
	      -v train_src="$$tato_train_src" -v train_trg="$$tato_train_trg" '\
	        BEGIN { \
	          n = split(projected, projected_pair, " "); \
	          evaluated_count = split(evaluated, evaluated_pair, " "); \
	          for (j = 1; j <= evaluated_count; j++) is_evaluated[evaluated_pair[j]] = 1; \
	        } \
	        { \
	          i = (NR - 1) % n; \
	          direction = projected_pair[i + 1]; \
	          if ($$1 == "" || $$2 == "" || \
	              $$1 ~ /^>>[^<]+<<[[:space:]]*$$/ || \
	              $$1 ~ /[[:space:]]\[NO_TRANSLATION\][[:space:]]*$$/ || \
	              $$2 == "[NO_TRANSLATION]") next; \
	          if (direction in is_evaluated && held[direction] < keep) { \
	            print $$1 >> (dev "." direction ".src"); \
	            print $$2 >> (dev "." direction ".trg"); \
	            held[direction]++; \
	          } else { \
	            print $$1 >> train_src; \
	            print $$2 >> train_trg; \
	          } \
	        }'; \
	}; \
	split_tatoeba_projection "$$tato_src" "$$tato_trg" "${MIX_PRIMARY_PAIRS}"; \
	if test -n "${MIX_SECONDARY_SRCLANGS}" && test -n "${MIX_SECONDARY_TRGLANGS}"; then \
	  "${MIX_HF_CONVERTER}" \
	    --repo-id "${TATOOEBA_HF_REPO}" \
	    --filename "${TATOOEBA_HF_FILE}" \
	    --revision main \
	    --cache-dir "${MIX_HF_CACHE_DIR}" \
	    --src-langs ${MIX_SECONDARY_SRCLANGS} \
	    --trg-langs ${MIX_SECONDARY_TRGLANGS} \
	    --corpus "${TATOOEBA_HF_CORPUS}" \
	    --nway-train-src "$$tato_secondary_src" \
	    --nway-train-trg "$$tato_secondary_trg" \
	    --nway-mode "${MIX_SECONDARY_MODE}" \
	    --nway-metadata "$$tato_secondary_meta" \
	    $(if ${MIX_SECONDARY_SOURCE_LANG},--nway-source-lang "${MIX_SECONDARY_SOURCE_LANG}") \
	    --keep-empty \
	    --skip-same-lang \
	    --overwrite; \
	  split_tatoeba_projection "$$tato_secondary_src" "$$tato_secondary_trg" \
	    "${MIX_SECONDARY_PAIRS}"; \
	fi; \
	for direction in ${MIX_EVAL_PAIRS}; do \
	  held=$$(wc -l < "$$tato_dev_base.$$direction.src" 2>/dev/null || echo 0); \
	  if test "$$held" -ne "${MIX_TATOOEBA_DEV_ROWS}"; then \
	    echo "Tatoeba direction $$direction has $$held valid holdout rows; expected ${MIX_TATOOEBA_DEV_ROWS}." >&2; \
	    exit 1; \
	  fi; \
	done; \
	if test "${MIX_USE_HIGH_QUALITY}" = 1; then \
	  "${MIX_HF_CONVERTER}" \
	    --repo-id "${HIGH_QUALITY_HF_REPO}" \
	    --filename "${HIGH_QUALITY_HF_FILE}" \
	    --revision main \
	    --cache-dir "${MIX_HF_CACHE_DIR}" \
	    --src-langs ${SRCLANGS} \
	    --trg-langs ${TRGLANGS} \
	    --corpus "${HIGH_QUALITY_HF_CORPUS}" \
	    --nway-train-src "$$hq_src" \
	    --nway-train-trg "$$hq_trg" \
	    --nway-mode "${HF_CSV_NWAY_MODE}" \
	    --nway-metadata "$$hq_meta" \
	    $(if ${HF_CSV_NWAY_SOURCE_LANG},--nway-source-lang "${HF_CSV_NWAY_SOURCE_LANG}") \
	    --skip-same-lang \
	    --nway-shuffle-buffer "${MIX_HQ_SHUFFLE_BUFFER}" \
	    --nway-shuffle-seed 1 \
	    --overwrite; \
	  cat "$$tato_train_src" "$$hq_src" > "${LOCAL_TRAIN_SRC}"; \
	  cat "$$tato_train_trg" "$$hq_trg" > "${LOCAL_TRAIN_TRG}"; \
	else \
	  mv "$$tato_train_src" "${LOCAL_TRAIN_SRC}"; \
	  mv "$$tato_train_trg" "${LOCAL_TRAIN_TRG}"; \
	fi; \
	printf "name\tstart\tend\n" > "${NAMED_VALIDATION_MANIFEST}"; \
	for direction in ${MIX_EVAL_PAIRS}; do \
	  src="$${direction%%-*}"; \
	  trg="$${direction#*-}"; \
	  if [[ "$$src" < "$$trg" ]]; then pair="$$src-$$trg"; \
	  else pair="$$trg-$$src"; fi; \
	  if test "$$src" != ar && test "$$trg" != ar; then \
	    corpus="bouquet-dev-$$trg"; \
	    bouquet_src="${DATADIR}/simple/$$corpus.$$pair.clean.$$src.gz"; \
	    bouquet_trg="${DATADIR}/simple/$$corpus.$$pair.clean.$$trg.gz"; \
	    test -s "$$bouquet_src" && test -s "$$bouquet_trg"; \
	    if test -e "${DEV_SRC}"; then start=$$(wc -l < "${DEV_SRC}"); \
	    else start=0; fi; \
	    if test -n "${MIX_BOUQUET_DEV_ROWS}"; then \
	      ${GZIP} -cd < "$$bouquet_src" | \
	        awk -v n="${MIX_BOUQUET_DEV_ROWS}" 'NR <= n' | \
	        sed "s/^/>>$$trg<< /" >> "${DEV_SRC}"; \
	      ${GZIP} -cd < "$$bouquet_trg" | \
	        awk -v n="${MIX_BOUQUET_DEV_ROWS}" 'NR <= n' >> "${DEV_TRG}"; \
	    else \
	      ${GZIP} -cd < "$$bouquet_src" | sed "s/^/>>$$trg<< /" >> "${DEV_SRC}"; \
	      ${GZIP} -cd < "$$bouquet_trg" >> "${DEV_TRG}"; \
	    fi; \
	    end=$$(wc -l < "${DEV_SRC}"); \
	    if test -n "${MIX_BOUQUET_DEV_ROWS}"; then \
	      test "$$((end - start))" -eq "${MIX_BOUQUET_DEV_ROWS}"; \
	    fi; \
	    printf "bouquet:%s-%s\t%s\t%s\n" "$$src" "$$trg" "$$start" "$$end" \
	      >> "${NAMED_VALIDATION_MANIFEST}"; \
	  fi; \
	  if test -e "${DEV_SRC}"; then start=$$(wc -l < "${DEV_SRC}"); \
	  else start=0; fi; \
	  cat "$$tato_dev_base.$$direction.src" >> "${DEV_SRC}"; \
	  cat "$$tato_dev_base.$$direction.trg" >> "${DEV_TRG}"; \
	  end=$$(wc -l < "${DEV_SRC}"); \
	  printf "tatooeba:%s-%s\t%s\t%s\n" "$$src" "$$trg" "$$start" "$$end" \
	    >> "${NAMED_VALIDATION_MANIFEST}"; \
	done; \
	test "$$(wc -l < "${LOCAL_TRAIN_SRC}")" -eq "$$(wc -l < "${LOCAL_TRAIN_TRG}")"; \
	test "$$(wc -l < "${DEV_SRC}")" -eq "$$(wc -l < "${DEV_TRG}")"; \
	cp "${DEV_SRC}" "${TEST_SRC}"; \
	cp "${DEV_TRG}" "${TEST_TRG}"; \
	rm -f "$$tato_src" "$$tato_trg" "$$tato_train_src" "$$tato_train_trg" \
	      "$$tato_secondary_src" "$$tato_secondary_trg" \
	      "$$tato_dev_base".*.src "$$tato_dev_base".*.trg \
	      "$$hq_src" "$$hq_trg"; \
	echo "prepared mixed train data and named BOUQuET/Tatoeba validation data"
