#!/bin/sh

PYTHON=/auto/nlg-05/chengham/anaconda3/envs/py37-old/bin/python
EVAL=nn.py

$PYTHON -W ignore $EVAL --model $1 \
  --task $2 \
  --output data/$1-$2-$3.rank \
  --embedder $3
