#!/bin/sh
LD_PRELOAD=./libtcmalloc.so.4.5.3 python3 ClassificationMLP.py 2>suc.log
