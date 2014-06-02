# define some Makefile variables for the compiler and compiler flags
# to use Makefile variables later in the Makefile: $()
#
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#  -pg   adds profiling information to the executable file
#  -m64  compile for 64-bit
CFLAGS = -g -Wall -m64 -fopenmp 
#-Wall  -fno-omit-frame-pointer 

# for C++ define  CC = g++
#  -m64  compile for 64 bits
CC = g++ -std=c++11 

# for c compiler
CCC = gcc

# archive (compile to lib*.a)
AR = ar -r -s

# define any directories containing header files other than /usr/include
INCLUDES  = -I ./
INCLUDES += -I ../

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS    = -L ./

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so:
LIBS = 

# define the cpp source files
SRCS  = Line3D.cpp Line3DTwoPoint.cpp Line3DPointDir.cpp LevenburgMaquart.cpp

# define the executable file 
TARGET = modelfitting

# Yuchen: The remove command.
# It is for Windoes only. Under Linux, ti will be something 
# like the following (which is not tested): 
#	rm -f
RM = rm 

all: $(CBLAS_OBJS) $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) $(CBLAS_OBJS) main.cpp -o $(TARGET) $(LFLAGS) $(LIBS)

# Yuchen: these following command will compile the other cpp files in the project
# For example, if there is a file SparseMatrix.cpp in the current directory, it will 
#   be compiled to SparseMatrix.o. That is equivalent to the following two lines of code. 
# SparseMatrix.o: SparseMatrix.cpp
#	$(CC) $(CFLAGS) $(INCLUDES) -c SparseMatrix.cpp -o SparseMatrix.o
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Yuchen: Removes all .o files and the excutable file, so that the next make rebuilds them
clean: 
	$(RM) *.o $(TARGET) analysis.dot analysis.png analysis.txt gmon.out

