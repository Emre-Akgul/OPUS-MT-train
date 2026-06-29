# Train Opus-MT models

This package includes scripts for training NMT models using MarianNMT and OPUS data for [OPUS-MT](https://github.com/Helsinki-NLP/Opus-MT). More details are given in the [Makefile](Makefile) but documentation needs to be improved. Also, the targets require a specific environment and right now only work well on the CSC HPC cluster in Finland.


## Pre-trained models

The subdirectory [models](https://github.com/Helsinki-NLP/Opus-MT-train/tree/master/models) contains information about pre-trained models that can be downloaded from this project. They are distribted with a [CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) license. [More pre-trained models](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/results/tatoeba-results-all.md) trained with the [OPUS-MT training pipeline](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/doc/TatoebaChallenge.md) are available from the [Tatoeba translation challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge) also under a [CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/) license.


## Quickstart

Setting up:

```
git clone https://github.com/Helsinki-NLP/OPUS-MT-train.git
git submodule update --init --recursive --remote
make install
```

Look into `lib/env.mk` and adust any settings that you need in your environment.
For CSC-users: adjust `lib/env/puhti.mk` and `lib/env/mahti.mk` to match yoursetup (especially the locations where Marian-NMT and other tools are installed and the CSC project that you are using).

Training a multilingual NMT model (Finnish and Estonian to Danish, Swedish and English):

```
make SRCLANGS="fi et" TRGLANGS="da sv en" train
make SRCLANGS="fi et" TRGLANGS="da sv en" eval
make SRCLANGS="fi et" TRGLANGS="da sv en" release
```

### Hugging Face CSV N-way training

For an aligned N-way CSV on Hugging Face, set the dataset repo, CSV file, and
language columns. The optimized direct path reads the CSV once and writes Marian
source/target training streams directly.

One source language to many targets:

```
export HF_TOKEN=...   # only needed for private Hugging Face repos

make \
  HF_CSV_REPO=TartarusXXX/tatooeba-synthetic \
  HF_CSV_FILE=data/train.csv \
  HF_CSV_NWAY_TRAIN=1 \
  HF_CSV_SKIP_PAIR_DATA=1 \
  HF_CSV_NWAY_MODE=one-to-many \
  HF_CSV_NWAY_SOURCE_LANG=en \
  SRCLANGS=en \
  TRGLANGS="tr kmr" \
  USE_TARGET_LABELS=1 \
  train
```

All directed language pairs from the same N-way table:

```
make \
  HF_CSV_REPO=TartarusXXX/tatooeba-synthetic \
  HF_CSV_FILE=data/train.csv \
  HF_CSV_NWAY_TRAIN=1 \
  HF_CSV_SKIP_PAIR_DATA=1 \
  HF_CSV_NWAY_MODE=all-to-all \
  SRCLANGS="en tr kmr" \
  TRGLANGS="en tr kmr" \
  USE_TARGET_LABELS=1 \
  SKIP_SAME_LANG=1 \
  train
```

If target labels should differ from CSV column names, pass mappings such as:

```
TARGET_LABEL_BY_TRGLANG="apc:ar arz:ar"
```

### Fine-tuning from a Hugging Face Marian model

Set `HF_MARIAN_MODEL_REPO` to a Hugging Face model repo id or URL. The Makefile
downloads the OPUS-MT/Marian artifacts into `${WORKHOME}/hf-models`, stages
stable filenames (`model.npz`, `vocab.yml`, `source.spm`, `target.spm`), reuses
the downloaded vocabulary and SentencePiece models, and starts Marian with
`--pretrained-model`.

```
export HF_TOKEN=...   # only needed for private Hugging Face repos

make \
  HF_MARIAN_MODEL_REPO=TartarusXXX/synthetic-en2n-marian \
  HF_CSV_REPO=TartarusXXX/tatooeba-synthetic \
  HF_CSV_FILE=data/train.csv \
  HF_CSV_NWAY_TRAIN=1 \
  HF_CSV_SKIP_PAIR_DATA=1 \
  HF_CSV_NWAY_MODE=one-to-many \
  HF_CSV_NWAY_SOURCE_LANG=en \
  SRCLANGS=en \
  TRGLANGS="tr kmr" \
  USE_TARGET_LABELS=1 \
  train
```

Useful model repo commands:

```
make HF_MARIAN_MODEL_REPO=TartarusXXX/synthetic-en2n-marian hf-marian-model
make HF_MARIAN_MODEL_REPO=TartarusXXX/synthetic-en2n-marian refresh-hf-marian-model
```

If automatic file detection is ambiguous, set one or more explicit filenames:

```
HF_MARIAN_MODEL_FILE=...best-bleu.npz
HF_MARIAN_VOCAB_FILE=...vocab.yml
HF_MARIAN_SOURCE_SPM_FILE=opus.src.spm32k-model
HF_MARIAN_TARGET_SPM_FILE=opus.trg.spm32k-model
```

### Named validation BLEU

By default Marian reports one aggregate validation score. To also log BLEU per
validation subset, enable named validation:

```
make MARIAN_NAMED_VALIDATION=1 train
```

The aggregate score remains in the normal Marian validation log and is used for
checkpointing. Per-subset scores are appended to:

```
${WORKDIR}/${MODEL}.${MODELTYPE}.valid${NR}.sets.tsv
```

With automatic dev set construction, the manifest is written to
`${DEV_SRC}.sets.tsv`. Named validation keeps the full dev set and disables dev
shuffling so the manifest line ranges stay aligned with Marian's hypotheses.

More information is available in the documentation linked below.


## Documentation

* [Installation and setup](https://github.com/Helsinki-NLP/Opus-MT-train/tree/master/doc/Setup.md)
* [Details about tasks and recipes](https://github.com/Helsinki-NLP/Opus-MT-train/tree/master/doc/README.md)
* [Information about back-translation](https://github.com/Helsinki-NLP/Opus-MT-train/tree/master/backtranslate/README.md)
* [Information about Fine-tuning models](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/finetune/README.md)
* [How to generate pivot-language-based translations](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/pivoting/README.md)



## Tutorials

* [Training low-resource models](https://github.com/Helsinki-NLP/Opus-MT-train/tree/master/doc/tutorials/low-resource.md)
* [How to train models for the Tatoeba MT Challenge](https://github.com/Helsinki-NLP/Opus-MT-train/tree/master/doc/TatoebaChallenge.md)


## References

Please, cite the following papers if you use OPUS-MT software and models:

```bibtex
@article{tiedemann2023democratizing,
  title={Democratizing neural machine translation with {OPUS-MT}},
  author={Tiedemann, J{\"o}rg and Aulamo, Mikko and Bakshandaeva, Daria and Boggia, Michele and Gr{\"o}nroos, Stig-Arne and Nieminen, Tommi and Raganato\
, Alessandro and Scherrer, Yves and Vazquez, Raul and Virpioja, Sami},
  journal={Language Resources and Evaluation},
  number={58},
  pages={713--755},
  year={2023},
  publisher={Springer Nature},
  issn={1574-0218},
  doi={10.1007/s10579-023-09704-w}
}

@InProceedings{TiedemannThottingal:EAMT2020,
  author = {J{\"o}rg Tiedemann and Santhosh Thottingal},
  title = {{OPUS-MT} — {B}uilding open translation services for the {W}orld},
  booktitle = {Proceedings of the 22nd Annual Conferenec of the European Association for Machine Translation (EAMT)},
  year = {2020},
  address = {Lisbon, Portugal}
 }
 ```


## Acknowledgements

None of this would be possible without all the great open source software including

* GNU/Linux tools
* [Marian-NMT](https://github.com/marian-nmt/)
* [eflomal](https://github.com/robertostling/eflomal)

... and many other tools like terashuf, pigz, jq, Moses SMT, fast_align, sacrebleu ...

We would also like to acknowledge the support by the [University of Helsinki](https://blogs.helsinki.fi/language-technology/), the [IT Center of Science CSC](https://www.csc.fi/en/home), the funding through projects in the EU Horizon 2020 framework ([FoTran](http://www.helsinki.fi/fotran), [MeMAD](https://memad.eu/), [ELG](https://www.european-language-grid.eu/), [HPLT](https://hplt-project.org)) and the contributors to the open collection of parallel corpora [OPUS](http://opus.nlpl.eu/).
