# Evaluation config for BOUQuET sentence-level dev.
#
# Covers the requested languages plus explicit Arabic dialects:
#   apc = Levantine Arabic
#   arz = Egyptian Arabic
#
# Standard Arabic is not included because the current BOUQuET sentence-level
# dev split does not publish arb_Arab.parquet.
#
# Prepare Moses files:
#   make -f configs/bouquet-dev-all-dialects/bouquet-dev.eval.mk -f Makefile bouquet-data
#
# Evaluate with this config after pointing it at a compatible multilingual
# model/work directory:
#   make -f configs/bouquet-dev-all-dialects/bouquet-dev.eval.mk -f Makefile eval

LANGS = tr en de es fr el bg ru ka hy fa ckb ur kmr apc arz

PYTHON ?= $(if $(wildcard work/venv-bouquet/bin/python),work/venv-bouquet/bin/python,python3)

SRCLANGS    = ${LANGS}
TRGLANGS    = ${LANGS}
LANGPAIRSTR = tr+en+de+es+fr+el+bg+ru+ka+hy+fa+ckb+ur+kmr+apc+arz-tr+en+de+es+fr+el+bg+ru+ka+hy+fa+ckb+ur+kmr+apc+arz

MODELTYPE = transformer
PRE       = simple
SUBWORDS  = spm

SKIP_SAME_LANG = 1
CLEAN_TRAINDATA_TYPE = clean
CLEAN_DEVDATA_TYPE   = clean
CLEAN_TESTDATA_TYPE  = clean

TESTSET_NAME = bouquet-dev
TESTSET      = bouquet-dev
KEEP_FULL_TESTSET = 1

BOUQUET_CORPUS_PREFIX = bouquet-dev
BOUQUET_CORPUS_MODES  = pair target
BOUQUET_LANGS = tur_Latn:tr eng_Latn:en deu_Latn:de spa_Latn:es fra_Latn:fr \
                ell_Grek:el bul_Cyrl:bg rus_Cyrl:ru kat_Geor:ka hye_Armn:hy \
                pes_Arab:fa ckb_Arab:ckb urd_Arab:ur kmr_Latn:kmr \
                apc_Arab:apc arz_Arab:arz

TESTSET_BY_TRGLANG = tr:bouquet-dev-tr en:bouquet-dev-en de:bouquet-dev-de \
                     es:bouquet-dev-es fr:bouquet-dev-fr el:bouquet-dev-el \
                     bg:bouquet-dev-bg ru:bouquet-dev-ru ka:bouquet-dev-ka \
                     hy:bouquet-dev-hy fa:bouquet-dev-fa ckb:bouquet-dev-ckb \
                     ur:bouquet-dev-ur kmr:bouquet-dev-kmr \
                     apc:bouquet-dev-apc arz:bouquet-dev-arz

GPUJOB_HPC_MEM = 8g
GPUJOB_SUBMIT =
