if ! unzip > /dev/null; then
    echo unzip does not exitst. installing unzip. 
    apt-get update
    sudo apt-get install unzip
else 
    echo unzip is already installed. skip unzip installation. 
fi

TEMP_DIR=temp

# -o overwrite the files if they already exists under $TEMP_DIR
unzip -o gtest-1.7.0.zip -d "$TEMP_DIR"

cd "$TEMP_DIR"/gtest-1.7.0

cmake ./

make

mkdir -p ../../../libs

cp ./libgtest.a ../../../libs/libgtest.a

cp ./libgtest_main.a ../../../libs/libgtest_main.a

cd ../../

rm -r -f "$TEMP_DIR"
