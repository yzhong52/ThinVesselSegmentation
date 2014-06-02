#!/bin/sh
####################################
#
# execuate all makefiles in ModelFitting, SparseMatrix, SparseMatrixCV， Vesselness
#
####################################

echo "\n"
echo "####################################"
echo "# Compiling Vesselness"
echo "####################################"
echo "\n"

cd ./Vesselness
make clean
make

echo "\n"
echo "####################################"
echo "# Compiling SparseMatrix"
echo "####################################"
echo "\n"

cd ../SparseMatrix
make clean
make

echo "\n"
echo "####################################"
echo "# Compiling SparseMatrixCV"
echo "####################################"
echo "\n"

cd ../SparseMatrixCV
make clean
make

echo "\n"
echo "####################################"
echo "# Compiling ModelFitting"
echo "####################################"
echo "\n"

cd ../ModelFitting
make clean
make

