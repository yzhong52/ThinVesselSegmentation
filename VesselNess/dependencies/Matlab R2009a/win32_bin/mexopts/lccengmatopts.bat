@echo off
rem LCCENGMATOPTS.BAT
rem
rem    Compile and link options used for building standalone engine or MAT programs
rem    with LCC C compiler 2.4
rem
rem    $Revision: 1.5.4.1 $  $Date: 2006/06/23 19:04:42 $
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************
set MATLAB=%MATLAB%
set PATH=%MATLAB%\sys\lcc\bin;%PATH%
set LCCMEX=%MATLAB%\sys\lcc\mex
set PERL="%MATLAB%\sys\perl\win32\bin\perl.exe"
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=lcc
set OPTIMFLAGS=-DNDEBUG
set DEBUGFLAGS=-g4
set COMPFLAGS=-c -Zp8 -I"%MATLAB%\sys\lcc\include" -noregistrylookup 
set NAME_OBJECT=-Fo

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\lcc
set LINKER=lcclnk
set LINKFLAGS=-tmpdir "%OUTDIR%." -L"%MATLAB%\sys\lcc\lib" -libpath "%LIBLOC%"
set LINKFLAGSPOST=libmx.lib libmat.lib libeng.lib
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=
set LINK_FILE=
set LINK_LIB= 
set NAME_OUTPUT=-o "%OUTDIR%%MEX_NAME%.exe"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=
set RC_LINKER=
