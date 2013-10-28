@echo off
rem MSVC60ENGMATOPTS.BAT
rem
rem    Compile and link options used for building standalone engine or MAT programs
rem    with Microsoft Visual C++ compiler version 6.0
rem
rem    $Revision: 1.5.2.4 $  $Date: 2007/02/02 23:11:09 $
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************
set MATLAB=%MATLAB%
set MSDevDir=%MSDevDir%
set VSINSTALLDIR=%MSDevDir%\..\..
set VCINSTALLDIR=%VSINSTALLDIR%\VC98
set PATH=%VCINSTALLDIR%\BIN;%VSINSTALLDIR%\Common\msdev98\bin;%PATH%
set INCLUDE=%VCINSTALLDIR%\INCLUDE;%VSINSTALLDIR%\Common\msdev98\MFC\INCLUDE;%VCINSTALLDIR%\ATL\INCLUDE;%INCLUDE%
set LIB=%VCINSTALLDIR%\LIB;%VCINSTALLDIR%\MFC\LIB;%LIB%
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=cl
set OPTIMFLAGS=-O2 -DNDEBUG
set DEBUGFLAGS=-Zi -Fd"%OUTDIR%%MEX_NAME%.pdb"
set COMPFLAGS=-c -Zp8 -G5 -W3 -EHs -nologo 
set NAME_OBJECT=/Fo

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\microsoft
set LINKER=link
set LINKFLAGS=/LIBPATH:"%LIBLOC%" libmx.lib libmat.lib libeng.lib /nologo /MACHINE:IX86 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/debug
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT="/out:%OUTDIR%%MEX_NAME%.exe"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=
set RC_LINKER=
