@echo off
rem LCCCOMPP.BAT
rem
rem    Compile and link options used for building MATLAB compiler
rem    applications with C Math Library with the LCC C compiler 
rem
rem    $Revision: 1.13.4.6 $  $Date $
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
set DLLCOMPFLAGS=-c -Zp8 -I"%MATLAB%\sys\lcc\include" -noregistrylookup 
set NAME_OBJECT=-Fo
rem ********************************************************************
rem Library creation command
rem ********************************************************************
set DLL_MAKEDEF=type %BASE_EXPORTS_FILE% | %PERL% -e "print \"LIBRARY %MEX_NAME%\nEXPORTS\n\"; while (<>) {print;}" > %LIB_NAME%.def
set DLL_MAKEDEF1=lcc %DLLCOMPFLAGS% "%MATLAB%\sys\lcc\mex\lccstub.c" -Fo%LIB_NAME%_stub.obj

rem ********************************************************************
rem Linker parameters
rem MATLAB_EXTLIB is set automatically by mex.bat
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\lcc
set LINKER=lcclnk
set LINKFLAGS=-tmpdir "%OUTDIR%." -L"%MATLAB%\sys\lcc\lib" -libpath "%LIBLOC%"
set LINKFLAGSPOST=mclmcrrt.lib
set DLLLINKFLAGS=-dll "%LIB_NAME%.def" %LINKFLAGS% %LIB_NAME%_stub.obj
set DLLLINKFLAGSPOST=%LINKFLAGSPOST%
set HGLINKFLAGS=
set HGLINKFLAGSPOST=sgl.lib
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=
set LINK_FILE=
set LINK_LIB= 
set NAME_OUTPUT=-o "%OUTDIR%%MEX_NAME%.exe"
set DLL_NAME_OUTPUT=-o "%OUTDIR%%MEX_NAME%.dll"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=
set RC_LINKER=

set POSTLINK_CMDS1="if exist %LIB_NAME%.def del %LIB_NAME%.def"
set POSTLINK_CMDS2="if exist %LIB_NAME%_stub.obj del %LIB_NAME%_stub.obj"
