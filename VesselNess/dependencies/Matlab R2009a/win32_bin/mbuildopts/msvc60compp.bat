@echo off
rem MSVC60COMPP.BAT
rem
rem    Compile and link options used for building MATLAB compiler programs
rem    with Microsoft Visual C++ compiler version 6.0 
rem
rem    $Revision: 1.18.4.11 $  $Date: 2008/05/27 18:20:53 $
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************
set MATLAB=%MATLAB%
set MSDevDir=%MSDevDir%
set VSINSTALLDIR=%MSDevDir%\..\..
set VCINSTALLDIR=%VSINSTALLDIR%\VC98
set PATH=%VCINSTALLDIR%\BIN;%VSINSTALLDIR%\Common\msdev98\bin;%MATLAB_BIN%;%PATH%
set INCLUDE=%VCINSTALLDIR%\INCLUDE;%VCINSTALLDIR%\MFC\INCLUDE;%VCINSTALLDIR%\ATL\INCLUDE;%INCLUDE%
set LIB=%VCINSTALLDIR%\LIB;%VCINSTALLDIR%\MFC\LIB;%LIB%
set PERL="%MATLAB%\sys\perl\win32\bin\perl.exe"
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=cl
set OPTIMFLAGS=-O2 -DNDEBUG
set DEBUGFLAGS=-Z7
set CPPOPTIMFLAGS=-O2 -DNDEBUG
set CPPDEBUGFLAGS=-Z7
set COMPFLAGS=-c -Zp8 -G5 -GX -W3 -nologo
set CPPCOMPFLAGS=-c -Zp8 -G5 -W3 -nologo -Zm500 -GX -MD -I"%MATLAB%\extern\include\cpp" -DMSVC -DIBMPC
set DLLCOMPFLAGS=-c -Zp8 -G5 -GX -W3 -nologo -I"%MATLAB%\extern\include\win32" -DMSVC -DIBMPC
set NAME_OBJECT=/Fo

rem ********************************************************************
rem Library creation commands creating import and export libraries
rem ********************************************************************
set DLL_MAKEDEF=type %BASE_EXPORTS_FILE% | %PERL% -e "print \"LIBRARY %MEX_NAME%.dll\nEXPORTS\n\"; while (<>) {print;}" > %DEF_FILE%

rem ********************************************************************
rem Linker parameters
rem MATLAB_EXTLIB is set automatically by mex.bat
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\microsoft
set LINKER=link
set LINKFLAGS=kernel32.lib user32.lib gdi32.lib advapi32.lib oleaut32.lib ole32.lib /LIBPATH:"%LIBLOC%" /nologo 
set LINKFLAGS=%LINKFLAGS% mclmcrrt.lib %MCR_DELAYLIB% %MCR_DELAYLOAD%
set CPPLINKFLAGS=
set DLLLINKFLAGS= %LINKFLAGS% /dll /implib:"%OUTDIR%%MEX_NAME%.lib" /def:%DEF_FILE%
set HGLINKFLAGS=sgl.lib
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/debug
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT="/out:%OUTDIR%%MEX_NAME%.exe"
set DLL_NAME_OUTPUT="/out:%OUTDIR%%MEX_NAME%.dll"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=rc /fo "%OUTDIR%%RES_NAME%.res"
set RC_LINKER= 

rem ********************************************************************
rem IDL Compiler
rem ********************************************************************
set IDL_COMPILER=midl /nologo /win32 /I "%MATLAB%\extern\include" 
set IDL_OUTPUTDIR= /out "%OUTDIRN%"
set IDL_DEBUG_FLAGS= /D "_DEBUG" 
set IDL_OPTIM_FLAGS= /D "NDEBUG" 
set POSTLINK_CMDS1="if exist %LIB_NAME%.def del %LIB_NAME%.def"
