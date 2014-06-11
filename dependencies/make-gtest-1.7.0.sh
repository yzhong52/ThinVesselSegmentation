if ! unzip > /dev/null; then
    echo unzip does not exitst. installing unzip. 
    apt-get update
    sudo apt-get install unzip
else 
    echo unzip is already installed. skip unzip installation. 
fi

TEMP_DIR=temp
DEST_DIR=../../../libs/gtest

# -o overwrite the files if they already exists under $TEMP_DIR
unzip -o gtest-1.7.0.zip -d "$TEMP_DIR"

cd "$TEMP_DIR"/gtest-1.7.0

cmake ./

make

mkdir -p "$DEST_DIR"

cp ./libgtest.a "$DEST_DIR"/libgtest.a

cp ./libgtest_main.a "$DEST_DIR"/libgtest_main.a


# -a : Preserve the specified attributes such as directory an file mode, ownership, timestamps, if possible additional attributes: context, links, xattr, all.
# -v : Explain what is being done.
# -r : Copy directories recursively.
cp -avr ./include "$DEST_DIR"/include

cd ../../

rm -r -f "$TEMP_DIR"
