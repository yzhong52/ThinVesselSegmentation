# Inlcude general makefile settings
include ../makefile-common.make

# define any directories containing header files other than /usr/include
INCLUDES += -I ./CBLAS

# define the cpp source files
SRCS = SparseMatrix.cpp SparseMatrixData.cpp SparseMatrix-lsover.cpp
SRCS_CBLAS = daxpy.c dcopy.c ddot.c dscal.c 

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

# Yuchen: For compiling CBLAS files. CBLAS is library for Basic Linear Algebra Subprograms
# The following files are downloaded from:
#  - http://www.netlib.org/clapack/cblas/
# They will be compile to CBLAS_ddot.o, CBLAS_daxpy.o, and etc. 
OBJS_CBLAS = $(SRCS_CBLAS:%.c=./obj/CBLAS_%.o) 

# define the executable file 
TARGET = ../libs/Release/libSparseMatrix.a

all: $(OBJS_CBLAS) $(OBJS)
	$(AR) $(TARGET) $(OBJS) $(OBJS_CBLAS)

# Yuchen: these following command will compile the other cpp files in the project
# For example, if there is a file SparseMatrix.cpp in the current directory, it will 
#   be compiled to SparseMatrix.o. That is equivalent to the following two lines of code. 
# SparseMatrix.o: SparseMatrix.cpp
#	$(CC) $(CFLAGS) $(INCLUDES) -c SparseMatrix.cpp -o SparseMatrix.o
./obj/%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# A pattern rule ‘%.o : %.c’ says how to make any file stem.o from another file stem.c
# Reference: http://www.gnu.org/software/make/manual/make.html#Pattern-Rules
# Yuchen: this following command will compile these following objects daxpy.o dscal.o ddot.o daxpy.o
./obj/CBLAS_%.o : CBLAS/%.c
	$(CCC) -m64 $(CFLAGS) $(INCLUDES) -c $< -o $@

# Yuchen: Removes all .o files and the excutable file, so that the next make rebuilds them
clean: 
	$(RM) ./obj/*.o $(TARGET)

