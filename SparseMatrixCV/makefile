# define some Makefile variables for the compiler and compiler flags
# to use Makefile variables later in the Makefile: $()
#
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#  -pg   adds profiling information to the executable file
#  -m64  compile for 64-bit
#  -Wno-comment disable warnings about multiline comments
CFLAGS = -Wall -m64 -fopenmp -Wno-comment -O2

# for C++ define  CC = g++
#  -m64  compile for 64 bits
CC = g++ -std=c++0x

# for c compiler
CCC = gcc

# archive (compile to lib*.a)
AR = ar -r -s

# define any directories containing header files other than /usr/include
INCLUDES  = -I ./
INCLUDES += -I ./CBLAS

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS  = -L ./
LFLAGS  = -L ../libs/

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so:
LIBS = libSparseMatrix.a

# define the cpp source files
SRCS = SparseMatrixCV-lsover.cpp SparseMatrixCV.cpp

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

# define the executable file 
TARGET = ../libs/libSparseMatrixCV.a

# Yuchen: The remove command.
# It is for Windoes only. Under Linux, ti will be something 
# like the following (which is not tested): 
#	rm -f
RM = rm -f

all: $(CBLAS_OBJS) $(OBJS)
	mkdir -p ../libs
	$(AR) $(TARGET) $(OBJS)

# Yuchen: these following command will compile the other cpp files in the project
# For example, if there is a file SparseMatrix.cpp in the current directory, it will 
#   be compiled to SparseMatrix.o. That is equivalent to the following two lines of code. 
# SparseMatrix.o: SparseMatrix.cpp
#	$(CC) $(CFLAGS) $(INCLUDES) -c SparseMatrix.cpp -o SparseMatrix.o
./obj/%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Yuchen: Removes all .o files and the excutable file, so that the next make rebuilds them
clean: 
	$(RM) ./obj/*.o $(TARGET)

