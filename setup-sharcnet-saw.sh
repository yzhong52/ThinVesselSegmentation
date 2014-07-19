# This is for testing on Sharcnet

# Change compiler
module unload intel/12.1.3
module load gcc/4.8.2

# Python
module load python/3.2.2


# OpenCV
module load opencv/2.4.9
export PKG_CONFIG_PATH=/opt/sharcnet/opencv/2.4.9/lib/pkgconfig
pkg-config --modversion opencv
pkg-config --cflags opencv
pkg-config --libs opencv

# Show all modules
module list

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

