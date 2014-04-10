# Important Notes by Yuchen Zhong,  yzhong.cs@gmail.com
# 1. This make file is only for 64-bit windows machine, tested under: 
#    - Windows 7
#    - 64 bit
# 2. It is only tested under Window 7 with MinGW-w64 
#    - link for MinGW-w64 http://sourceforge.net/projects/mingw-w64/
# 3. Some of the code for this make file is borrowed from:
#    - http://www.cs.swarthmore.edu/~newhall/unixhelp/howto_makefiles.html




# define some Makefile variables for the compiler and compiler flags
# to use Makefile variables later in the Makefile: $()
#
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#  -pg   adds profiling information to the executable file
#
CFLAGS = -g -pg #-Wall 

# for C++ define  CC = g++
#  -m64  compile for 64 bits
CC = g++ -std=c++11 -m64

# define any directories containing header files other than /usr/include
#
INCLUDES = -I ./
INCLUDES += -I "../dependencies/OpenCV 2.4.8/include"
INCLUDES += -I "../VesselNess"
INCLUDES += -I "gco-v3.0"


# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS = -L ./

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so:
LIBS = -lopencv_core243.dll

# define the cpp source files
SRCS = SparseMatrix.cpp 

# define the C object files 
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
# $(SRCS:.cpp=.o) 
OBJS = $(SRCS:%.cpp=%.o) 

# Yuchen: For compiling CBLAS files. CBLAS is library for Basic Linear Algebra Subprograms
# The following files are downloaded from:
#  - http://www.netlib.org/clapack/cblas/
# They will be compile to CBLAS_ddot.o, CBLAS_daxpy.o, and etc. 
CBLAS_SRCS = ddot.c daxpy.c dscal.c dcopy.c
CBLAS_OBJS = $(CBLAS_SRCS:%.c=CBLAS_%.o) 

# define the executable file 
TARGET = executable.exe

# Yuchen: The remove command.
# It is for Windoes only. Under Linux, ti will be something 
# like the following (which is not tested): 
#	rm -f
RM = del 

all: $(CBLAS_OBJS) $(OBJS)
	@echo.
	@echo Compile finished. Linking bein...
	@echo $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) $(CBLAS_OBJS) main.cpp -o $(TARGET) $(LFLAGS) $(LIBS)


# Yuchen: This is for profiling
prof: 
	$(TARGET) $(SRCS)
	gprof $(TARGET) gmon.out > analysis.txt
	profiling_analysis.txt



# A pattern rule ‘%.o : %.c’ says how to make any file stem.o from another file stem.c
# Reference: http://www.gnu.org/software/make/manual/make.html#Pattern-Rules
# Yuchen: this following command will compile these following objects daxpy.o dscal.o ddot.o daxpy.o
CBLAS_%.o : SparseMatrix/cblas/%.c
	gcc -m64 $(CFLAGS) $(INCLUDES) -c $< -o $@
# Yuchen: these following command will compile the other cpp files in the project
# For example, if there is a file SparseMatrix.cpp in the current directory, it will 
#   be compiled to SparseMatrix.o. That is equivalent to the following two lines of code. 
# SparseMatrix.o: SparseMatrix.cpp
#	$(CC) $(CFLAGS) $(INCLUDES) -c SparseMatrix.cpp -o SparseMatrix.o
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

SparseMatrix.o: SparseMatrix/SparseMatrix.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@



# Yuchen: Removes all .o files and the excutable file, so that the next make rebuilds them
clean: 
	$(RM) *.o $(TARGET)

