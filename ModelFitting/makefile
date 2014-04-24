# Important Notes by Yuchen Zhong,  yzhong.cs@gmail.com
# 1. This make file is only for 64-bit windows machine, tested under: 
#    - Windows 7
#    - 64 bit
# 2. It is only tested under Window 7 with MinGW-w64 
#    - link for MinGW-w64 http://sourceforge.net/projects/mingw-w64/
# 3. Some of the code for this make file is borrowed from:
#    - http://www.cs.swarthmore.edu/~newhall/unixhelp/howto_makefiles.html

# Yuchen: if you are using mingw32, do the following:
# 1. compile and link the project
# mingw32-make
# 2. profile the program
# mingw32-make prof
# 3. show result as txt
# mingw32-make prof-show-txt
# 4. show rsult as png image (the png file will be located under the same folder)
# mingw32-make prof-show-graph


# define some Makefile variables for the compiler and compiler flags
# to use Makefile variables later in the Makefile: $()
#
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#  -pg   adds profiling information to the executable file
#
CFLAGS = -g -pg 
#-Wall  -fno-omit-frame-pointer 

# for C++ define  CC = g++
#  -m64  compile for 64 bits
CC = g++ -std=c++11 -m64

# define any directories containing header files other than /usr/include
#
INCLUDES  = -I ./
INCLUDES += -I ../
INCLUDES += -I "../dependencies/opencv2.4.8/include"
INCLUDES += -I "../VesselNess"
INCLUDES += -I "gco-v3.0"


# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS  = -L ./
LFLAGS += -L ../dependencies/opencv2.4.8/x64-mingw32-w64


# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so:
LIBS = -lopencv_core248.dll

# define the cpp source files
SRCS  = SparseMatrix.cpp Line3D.cpp Line3DTwoPoint.cpp Line3DPointDir.cpp LevenburgMaquart.cpp
SRCS += Timer.cpp

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

TEXTEDITOR = "C:/Program Files/Sublime Text 2/sublime_text.exe"

all: $(CBLAS_OBJS) $(OBJS)
	@echo.
	@echo Compile finished. Linking bein...
	@echo $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJS) $(CBLAS_OBJS) main.cpp -o $(TARGET) $(LFLAGS) $(LIBS)


# Yuchen: The following s is for profiling
# Genereate the analysis file gprof.txt
# Yuchen: This is for profiling
prof: 
	$(TARGET) LevenburgMaquart.cpp
	gprof $(TARGET) gmon.out > analysis.txt
#prof: 
#	$(TARGET) LevenburgMaquart.cpp
#	gprof $(TARGET) gmon.out > analysis.txt
#	gprof executable.exe gmon.out > analysis.txt
	

# Show the txt files
prof-show-txt: 
	$(TEXTEDITOR) analysis.txt

# Analysise the txt file and gerate call graph
# In order to run the follow script 'prof-show-graph', you have to install the following:
#  - python 3.3 (official tested), or python 2.7 (official tested), or Python 3.4 (tested by Yuchen)
#  - GraphViz 2.26.3 (offical tested), or graphviz-2.38 (tested by Yuchen)
#  - make sure gprof2dot.py is in the current directory; otherwise, download it from https://code.google.com/p/jrfonseca/wiki/Gprof2Dot
#  - run scritp 'prof' in advance in order to get file 'analysis.txt'
prof-show-graph: 
	python gprof2dot.py -n0.5 -s analysis.txt > analysis.dot 
	dot -Tpng analysis.dot -Gcharset=latin1 -o analysis.png
	@echo Done... Image has been saved to analysis.png

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

Timer.o: ../Timer.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Yuchen: Removes all .o files and the excutable file, so that the next make rebuilds them
clean: 
	$(RM) *.o $(TARGET) analysis.dot analysis.png analysis.txt gmon.out

