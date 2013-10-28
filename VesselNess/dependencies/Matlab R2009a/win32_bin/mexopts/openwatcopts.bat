rem @echo off
rem OPENWATCOPTS.BAT
rem
rem    Compile and link options used for building MEX-files with
rem    the Open WATCOM C compiler
rem
rem StorageVersion: 1.0
rem C++keyFileName: OPENWATCOPTS.BAT
rem C++keyName: Open WATCOM C/C++ 
rem C++keyManufacturer: Sybase
rem C++keyVersion: 1.7
rem C++keyLanguage: C++
rem 
rem    $Revision: 1.1.6.2 $  $Date: 2007/11/07 17:44:08 $
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************
set MATLAB=%MATLAB%
set WATCOM=%WATCOM%
set PATH=%WATCOM%\BINNT;%WATCOM%\BINW;%PATH%
set INCLUDE=%WATCOM%\H;%WATCOM%\mfc\include;%WATCOM%\H\nt;%INCLUDE%
set LIB=%WATCOM%\LIB386\nt;%WATCOM%\LIB386;%LIB%
set EDPATH=%WATCOM%\eddat
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=wcl386
set COMPFLAGS=-c -bd -5s -e25 -ei -fpi87 -zp8 -zq -fr# -DMATLAB_MEX_FILE
set OPTIMFLAGS=-ox -DNDEBUG
set DEBUGFLAGS=-d2
set NAME_OBJECT=/fo#

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\watcom
set LINKER=wlink
set LINKFLAGS=system nt_dll export %ENTRYPOINT% option caseexact libpath %LIBLOC% library libmx.lib, libmex.lib, libmat.lib, user32.lib
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=debug all
set LINK_FILE=file
set LINK_LIB=library 
set NAME_OUTPUT=name %OUTDIR%%MEX_NAME%%MEX_EXT%
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=
set RC_LINKER=wrc /q  /fo=%OUTDIR%mexversion.res

