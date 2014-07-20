# This file contains important/usefull commands for compiling and running the code on Sharcnet.


###############################################
# Login to Sharcnet
###############################################
ssh saw.sharcnet.ca -l username


###############################################
# Clone the project
###############################################
git clone https://yzhong52@bitbucket.org/yzhong52/thin-structure-segmentation.git




###############################################
# Set up the compiling environment. 
###############################################

# Change compiler
module unload intel/12.1.3
module load gcc/4.8.2

# Python
# Please refer to 'Python Configuration lod.sh'

# OpenCV
module load opencv/2.4.9
export PKG_CONFIG_PATH=/opt/sharcnet/opencv/2.4.9/lib/pkgconfig
pkg-config --modversion opencv
pkg-config --cflags opencv
pkg-config --libs opencv

# Show all modules
module list





###############################################
# Copy File from to Sharcnet using 'scp'
###############################################
scp <filename> <username>@saw.sharcnet.ca:

example: 
# copy a data file (with its readme file) to a excutable bin folder on sharcnet
scp data15.et.vn_sig data15.et.vn_sig.readme.txt yzhong52@saw.sharcnet.ca:/../../scratch/yzhong52/thin-structure-segmentation/bin
# copy the output directory back to a local computer 
scp -r yzhong52@saw.sharcnet.ca:/../../scratch/yzhong52/thin-structure-segmentation/bin ./from-sharcnet



###############################################
# Submitting a job on sharcnet
###############################################
sqsub -q <Type of program> -n <number of cpus cores> -N <number of nodes> -o <output log file> -r <maximum run time> ./modelfitting
# Note: For example, on saw (sharcnet cluster name) there are 8 cores per node. 
sqsub --mpp=6g -q threaded -n 8 -N 1 -o modelfitting.log -r 5h ./modelfitting




