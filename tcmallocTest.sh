#!/bin/sh
LIB="../tcmalloc/bazel-bin/tcmalloc/libtcmalloc.so"
LD_PRELOAD=$LIB python3 ClassificationMLP.py 2>err.log
