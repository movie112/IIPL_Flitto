#!/bin/bash
set -e

SRC_LANG=$1
TGT_LANG=$2
TEST_PREFIX=$3
DEST_DIR=$4
DICT_DIR=$5

mkdir -p $DEST_DIR

python -m fairseq_cli.preprocess \
  --source-lang $SRC_LANG \
  --target-lang $TGT_LANG \
  --testpref $TEST_PREFIX \
  --destdir $DEST_DIR \
  --srcdict $DICT_DIR/dict.$SRC_LANG.txt \
  --tgtdict $DICT_DIR/dict.$TGT_LANG.txt \
  --workers 8
 