# README
Tensorflow tests come from https://github.com/whybfq/tensorflow2.git

## Plateform

Linux System

Python3

Tensorflow_gpu2

## Some files

libtcmalloc.so is build by tcmalloc source code

libtcmalloc.so.4.5.3 is extracted from google-perftools

err.log is the error message when running tensorflow tests linked with libtcmalloc.so

## To test libtcmalloc.so

./tcmallocTest.sh (error, will try to new a large memory and crash)

## To test libtcmalloc.so.4.5.3

./releaseTcmallocTest.sh (This makes me believe that Tensorflow can be run by linking)