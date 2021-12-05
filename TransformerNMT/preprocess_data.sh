# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# set -e

#
# Data preprocessing configuration
#

CODES=60000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fastBPE/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# Download and install tools
mkdir -p $TOOLS_PATH

# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download fastBPE
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"

# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR/fastBPE
  g++ -std=c++11 -pthread -O3 main.cc -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"

cd ..

DATA_PATH=$PWD/data

MONO_PATH1=$DATA_PATH/yelp
MONO_PATH2=$DATA_PATH/shakespeare
MONO_PATH3=$DATA_PATH/sr_bos

# Training files

SRC_RAW1=$MONO_PATH1/train_1.txt
SRC_RAW2=$MONO_PATH2/train_1.txt
SRC_RAW3=$MONO_PATH3/train_1.spm32000.txt

TGT_RAW1=$MONO_PATH1/train_0.txt
TGT_RAW2=$MONO_PATH2/train_0.txt
TGT_RAW3=$MONO_PATH3/train_0.spm32000.txt

# Vocab Files

CONCAT_BPE1=$MONO_PATH1/all
CONCAT_BPE2=$MONO_PATH2/all
CONCAT_BPE3=$MONO_PATH3/all

SRC_VOCAB1=$MONO_PATH1/src_vocab
SRC_VOCAB2=$MONO_PATH2/src_vocab
SRC_VOCAB3=$MONO_PATH3/src_vocab

TGT_VOCAB1=$MONO_PATH1/tgt_vocab
TGT_VOCAB2=$MONO_PATH2/tgt_vocab
TGT_VOCAB3=$MONO_PATH3/tgt_vocab

FULL_VOCAB1=$MONO_PATH1/full_vocab
FULL_VOCAB2=$MONO_PATH2/full_vocab
FULL_VOCAB3=$MONO_PATH3/full_vocab

# Test and Valid files

SRC_TEST1=$MONO_PATH1/test_1.txt
SRC_TEST2=$MONO_PATH2/test_1.txt
SRC_TEST3=$MONO_PATH3/test_1.spm32000.txt

TGT_TEST1=$MONO_PATH1/test_0.txt
TGT_TEST2=$MONO_PATH2/test_0.txt
TGT_TEST3=$MONO_PATH3/test_0.spm32000.txt

SRC_VALID1=$MONO_PATH1/dev_1.txt
SRC_VALID2=$MONO_PATH2/dev_1.txt
SRC_VALID3=$MONO_PATH3/dev_1.spm32000.txt

TGT_VALID1=$MONO_PATH1/dev_0.txt
TGT_VALID2=$MONO_PATH2/dev_0.txt
TGT_VALID3=$MONO_PATH3/dev_0.spm32000.txt

echo "Extracting vocabulary..."
cat $SRC_RAW1 | $FASTBPE getvocab - >$SRC_VOCAB1
cat $SRC_RAW2 | $FASTBPE getvocab - >$SRC_VOCAB2
cat $SRC_RAW3 | $FASTBPE getvocab - >$SRC_VOCAB3

cat $TGT_RAW1 | $FASTBPE getvocab - >$TGT_VOCAB1
cat $TGT_RAW2 | $FASTBPE getvocab - >$TGT_VOCAB2
cat $TGT_RAW3 | $FASTBPE getvocab - >$TGT_VOCAB3

echo "Creating Full Vocabularies"

cat $SRC_RAW1 $TGT_RAW1 | $FASTBPE getvocab - >$FULL_VOCAB1
cat $SRC_RAW2 $TGT_RAW1 | $FASTBPE getvocab - >$FULL_VOCAB2
cat $SRC_RAW3 $TGT_RAW1 | $FASTBPE getvocab - >$FULL_VOCAB3

echo "Binarizing Training Data..."
python3 $UMT_PATH/preprocess.py $FULL_VOCAB1 $SRC_RAW1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB2 $SRC_RAW2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB3 $SRC_RAW3

python3 $UMT_PATH/preprocess.py $FULL_VOCAB1 $TGT_RAW1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB2 $TGT_RAW2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB3 $TGT_RAW3

echo "Binarizing Validation Data..."
python3 $UMT_PATH/preprocess.py $FULL_VOCAB1 $SRC_TEST1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB2 $SRC_TEST2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB3 $SRC_TEST3

python3 $UMT_PATH/preprocess.py $FULL_VOCAB1 $TGT_TEST1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB2 $TGT_TEST2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB3 $TGT_TEST3

python3 $UMT_PATH/preprocess.py $FULL_VOCAB1 $SRC_VALID1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB2 $SRC_VALID2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB3 $SRC_VALID3

python3 $UMT_PATH/preprocess.py $FULL_VOCAB1 $TGT_VALID1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB2 $TGT_VALID2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB3 $TGT_VALID3

cat $SRC_RAW1 $TGT_RAW1 | shuf > $CONCAT_BPE1
cat $SRC_RAW2 $TGT_RAW2 | shuf > $CONCAT_BPE2
cat $SRC_RAW3 $TGT_RAW3 | shuf > $CONCAT_BPE3

echo "Training fastText on $CONCAT_BPE..."
$FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 256 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE1 -output $CONCAT_BPE1".256"
$FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 256 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE2 -output $CONCAT_BPE2".256"
$FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 256 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE3 -output $CONCAT_BPE3".256"
