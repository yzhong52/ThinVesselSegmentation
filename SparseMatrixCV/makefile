# Inlcude general makefile settings
include ../makefile-common.make

# define any directories containing header files other than /usr/include
INCLUDES = -I ./test
# For testing
INCLUDES += -I ../libs/gtest/include

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS  = -L ../libs/Release
# for gtest
LFLAGS += -L ../libs/gtest
# for opencv
LFLAGS += `pkg-config --libs opencv`

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so):
LIBS = -lSparseMatrix -lgtest

# define the cpp source files
SRCS = SparseMatrixCV.cpp
SRCS_TEST = SparseMatrixCVTest.cpp test.cpp

# define the C object files 
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
# $(SRCS:.cpp=.o) 
OBJS = $(SRCS:%.cpp=./obj/%.o) 
OBJS_TEST = $(SRCS_TEST:%.cpp=obj/test_%.o) 

# define the executable file 
TARGET = ../libs/Release/libSparseMatrixCV.a
TARGET_TEST = bin/SparseMatrixCV-test

all: $(OBJS)
	$(AR) $(TARGET) $(OBJS)

test: make-dir $(OBJS) $(OBJS_TEST) 
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCLUDES) $(OBJS) $(OBJS_TEST) -o $(TARGET_TEST) $(LFLAGS) $(LIBS)
	$(TARGET_TEST)

make-dir: 
	mkdir -p bin
	mkdir -p obj/test

# Yuchen: these following command will compile the other cpp files in the project
# For example, if there is a file SparseMatrix.cpp in the current directory, it will 
#   be compiled to SparseMatrix.o. That is equivalent to the following two lines of code. 
# SparseMatrix.o: SparseMatrix.cpp
#	$(CC) $(CFLAGS) $(INCLUDES) -c SparseMatrix.cpp -o SparseMatrix.o
obj/%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

obj/test_%.o: test/%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Yuchen: Removes all .o files and the excutable file, so that the next make rebuilds them
clean: 
	$(RM) -r bin
	$(RM) -r obj
	$(RM) $(TARGET)
	$(RM) $(TARGET_TEST)

