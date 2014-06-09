# define some Makefile variables for the compiler and compiler flags
# to use Makefile variables later in the Makefile: $()
#
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#  -pg   adds profiling information to the executable file
#  -m64  compile for 64-bit
#  -Wno-comment disable warnings about multiline comments
CFLAGS  = -Wall -m64 -fopenmp -Wno-comment -std=c++0x -O2 -Wl,--no-as-needed
# Add OpenCV include path
CFLAGS +=`pkg-config --cflags opencv`

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS  = -L ./
LFLAGS += `pkg-config --libs opencv` -Wl,--no-as-needed -pthread

# for C++ define  CC = g++
#  -m64  compile for 64 bits
CC = g++ 

# for c compiler
CCC = gcc

# archive (compile to lib*.a)
AR = ar -r -s

# Yuchen: The remove command.
# It is for Windoes only. Under Linux, ti will be something 
# like the following (which is not tested): 
#	rm -f
RM = rm -f
