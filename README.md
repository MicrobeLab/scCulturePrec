# scCulturePrec

An easy-to-use Python package implementing the machine learning framework presented in "Precision culturomics enabled by unlabeled single-cell morphology and Raman spectra"

## Installation

`scCulturePrec` is easily installed from the GitHub repository:

    git clone https://github.com/MicrobeLab/scCulturePrec.git
    cd scCulturePrec
    pip install -e .

## Usage

Use `scCulturePrec -h` to print help message:

    usage: scCulturePrec [-h] {create-pair,elastic-net,dl-model} ...

    positional arguments:
      {create-pair,elastic-net,dl-model}
        create-pair         Create paired-samples for training
        elastic-net         Elastic net for distances to reference
        dl-model            Siamese network: training, encoding feature vectors,
                            matching (default)

### (1) Positive and negative sample pairs

Use `scCulturePrec create-pair -h` to print help message:

    usage: scCulturePrec create-pair [-h] [--npy-fns NPY_FNS]
                                     [--half-num HALF_NUM] [--out OUT]
                                     [--block-size BLOCK_SIZE]

    optional arguments:
      -h, --help            show this help message and exit
      --npy-fns NPY_FNS     A text file with one .npy file name per line
      --half-num HALF_NUM   Total number of positive/negative sample pairs
      --out OUT             Output prefix
      --block-size BLOCK_SIZE
                            Number of sample pairs in each output block

Input data are provided as NumPy arrays, with each array containing samples from one taxon, shaped as `[number_of_samples, number_of_features]`. The morphological features and spectral features are concatenated along the second axis of the array.

### (2) Deep neural network

Use `scCulturePrec dl-model -h` to print help message:

    usage: scCulturePrec dl-model [-h] [--train] [--encode]
                                  [--batch-size BATCH_SIZE] [--fn FN]
                                  [--embed-size-spectra EMBED_SIZE_SPECTRA]
                                  [--embed-size-morphol EMBED_SIZE_MORPHOL]
                                  [--in-type {morphol,spectra,both}]
                                  [--num-morphol NUM_MORPHOL]
                                  [--input-dir INPUT_DIR] [--lr LR] [--ckpt CKPT]
                                  [--margin MARGIN] [--weight WEIGHT]
                                  [--feat-db FEAT_DB] [--out OUT]

    optional arguments:
      -h, --help            show this help message and exit
      --train               Training Siamese network
      --encode              Encoding feature vectors
      --batch-size BATCH_SIZE
                            Batch size
      --fn FN               Input filename
      --embed-size-spectra EMBED_SIZE_SPECTRA
                            Embedding size for spectral model
      --embed-size-morphol EMBED_SIZE_MORPHOL
                            Embedding size for morphological model
      --in-type {morphol,spectra,both}
                            morphol/spectra/both
      --num-morphol NUM_MORPHOL
                            Number of morphological features
      --input-dir INPUT_DIR
                            Input directory
      --lr LR               Learning rate
      --ckpt CKPT           Prefix of checkpoint files
      --margin MARGIN       Margin in contrastive loss
      --weight WEIGHT       /path/to/weights.pth
      --feat-db FEAT_DB     /path/to/ref_feature_map.h5
      --out OUT             Output prefix

The module operates in matching mode by default, which involves obtaining distances to reference samples. To switch to training and encoding modes, use the `--train` and `--encode` options.

### (3) Elastic net model

Use `scCulturePrec elastic-net -h` to print help message:

    usage: scCulturePrec elastic-net [-h] [--dist-pos DIST_POS]
                                     [--dist-neg DIST_NEG]
                                     [--model-file MODEL_FILE]
                                     [--dist-new DIST_NEW] [--pred-out PRED_OUT]
                                     [--train]

    optional arguments:
      -h, --help            show this help message and exit
      --dist-pos DIST_POS   Input distances of positive samples (.txt)
      --dist-neg DIST_NEG   Input distances of negative samples (.txt)
      --model-file MODEL_FILE
                            Elastic net model file (.pkl)
      --dist-new DIST_NEW   Input distances of new samples (.txt)
      --pred-out PRED_OUT   Predictions (.csv)
      --train               Training elastic net

The module operates in prediction mode by default. To switch to training mode, use the `--train` option. Input distance files contain tables of distances between new samples and reference samples, formatted as `[number_of_new_samples, number_of_reference_samples]`.

## Other Resources

Bugs and difficulties in using `scCulturePrec` are welcome on [the issue tracker](https://github.com/MicrobeLab/scCulturePrec/issues).

Example data sets and models are available [here]().
