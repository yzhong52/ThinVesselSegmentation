#######################################################
# The following is for using Python to display a plot
#######################################################

# Install python3
sudo apt-get python3

# Install pip -- a tool for installing and managing Python packages
sudo apt-get install python3-pip

# Install libpng - the official PNG reference library
sudo apt-get install libpng-dev

# Install FreeTyep -- a freely available software library to render fonts
sudo apt-get install libfreetype6-dev

# Install matplotlib -- a python 2D plotting library
sudo pip3 install matplotlib



#######################################################
# Install a Python without root account
#######################################################
# Do the following only if you want to use 'send_email.h" but don't have root access. 
# Otherwise you can just install python using sth similar to the following
# sudo apt-get install python3-dev

# Step 1. Install ssl-dev (This step is not necessary for sharcnet). 
sudo apt-get install libssl-dev

# Step 2. Download Python. 
# Note: If you upgrade the Python library to any library other than the following, you may have to update the makefile as well. Please refer to the following reference in terms of what to change. 
# Reference: http://stackoverflow.com/questions/8282231/ubuntu-i-have-python-but-gcc-cant-find-python-hmake
wget https://www.python.org/ftp/python/3.4.1/Python-3.4.1.tgz --no-check-certificate

# Step 3. Unzip it. 
tar -zxvf Python-3.4.1.tgz

# Step 4. Compile it. 
cd Python-3.4.1
./configure --prefix=/scratch/yzhong52/python --enable-unicode=ucs4
# Replace the above line to the following if you are not using Sharcnet. Please be aware that you need to modify the install path. 
# ./configure --prefix=/home/yzhong52/Desktop/python --enable-unicode=ucs4

# Step 5. Make and install the library. 
make
make install


