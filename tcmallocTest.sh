#!/bin/sh
LD_PRELOAD=./libtcmalloc.so python3 ClassificationMLP.py 2>err.log
