@echo off
rem OPENWATCENGMATOPTS.BAT
rem
rem    Compile and link options used for building MAT and engine standalone files
rem
rem    $Revision: 1.1.6.1 $  $Date: 2007/09/27 22:06:46 $
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************
set MATLAB=%MATLAB%
set WATCOM=%WATCOM%
set PATH=%WATCOM%\BINNT;%WATCOM%\BINW;%PATH%
set INCLUDE=%WATCOM%\H;%WATCOM%\H\NT;%INCLUDE%
set LIB=%WATCOM%\LIB386;%WATCOM%\LIB386\NT;%LIB%
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=wcl386
set COMPFLAGS= -c -bc -e25 -fpi87 -5s -zp8 -ei -bm -fr= -zq 
set OPTIMFLAGS=-ox -DNDEBUG
set DEBUGFLAGS=-d2
set NAME_OBJECT=-fo=

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\watcom
set LINKER=wlink
set LINKFLAGS=format windows nt option quiet libpath %LIBLOC% library libmx.lib, libmat.lib, libeng.lib, user32.lib, kernel32.lib
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=debug all
set LINK_FILE=file
set LINK_LIB=library  
set NAME_OUTPUT=name %OUTDIR%%MEX_NAME%.exe
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=
set RC_LINKER=
