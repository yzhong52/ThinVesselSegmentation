#!/bin/sh
####################################
#
# execuate all makefiles in ModelFitting, SparseMatrix, SparseMatrixCV， Vesselness
#
####################################

cd ./Vesselness
make clean
make

cd ../SparseMatrix
make clean
make

cd ../SparseMatrixCV
make clean
make

cd ../ModelFitting
make clean
make

