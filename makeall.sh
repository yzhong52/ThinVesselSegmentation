#!/bin/sh
####################################
#
# execuate all makefiles in ModelFitting, SparseMatrix, SparseMatrixCV， Vesselness
#
####################################

mkdir -p bin libs

echo ""
echo "####################################"
echo "# Compiling Vesselness"
echo "####################################"
echo ""

cd ./Vesselness
make clean
mkdir -p bin obj
make

echo ""
echo "####################################"
echo "# Compiling SparseMatrix"
echo "####################################"
echo ""

cd ../SparseMatrix
make clean
mkdir -p bin obj
make

echo ""
echo "####################################"
echo "# Compiling SparseMatrixCV"
echo "####################################"
echo ""

cd ../SparseMatrixCV
make clean
mkdir -p bin obj
make

echo ""
echo "####################################"
echo "# Compiling ModelFitting"
echo "####################################"
echo ""

cd ../ModelFitting
make clean
mkdir -p bin obj
make


