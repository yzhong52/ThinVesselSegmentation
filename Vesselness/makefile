# Inlcude general makefile settings
include ../makefile-common.make

# define any directories containing header files other than /usr/include
INCLUDES += -I ../core

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so:
# LIBS  = libSparseMatrixCV.a libSparseMatrix.a
# LIBS += -lGL -lGLU -lglut -lGLEW

# define the cpp source files
SRCS  = VesselnessTypes.cpp VesselDetector.cpp ImageProcessing-Vesselness.cpp

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
TARGET = ../bin/vesselness

all: $(CBLAS_OBJS) $(OBJS)
	mkdir -p ../bin
	mkdir -p ./obj
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCLUDES) $(OBJS) main_sharcnet.cpp -o $(TARGET) $(LFLAGS) $(LIBS)

run:  
	$(TARGET)

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

