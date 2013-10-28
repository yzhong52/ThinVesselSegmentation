@echo off
rem MSVC60OPTS.BAT
rem
rem    Compile and link options used for building MEX-files
rem    using the Microsoft Visual C++ compiler version 6.0 
rem
rem StorageVersion: 1.0
rem C++keyFileName: MSVC60OPTS.BAT
rem C++keyName: Microsoft Visual C++
rem C++keyManufacturer: Microsoft
rem C++keyVersion: 6.0
rem C++keyLanguage: C++
rem
rem    $Revision: 1.11.4.12 $  $Date: 2008/05/27 18:21:02 $
rem    Copyright 1984-2008 The MathWorks, Inc.
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************
set MATLAB=%MATLAB%
set MSDevDir=%MSDevDir%
set VSINSTALLDIR=%MSDevDir%\..\..
set VCINSTALLDIR=%VSINSTALLDIR%\VC98
set PATH=%VCINSTALLDIR%\BIN;%VSINSTALLDIR%\Common\msdev98\bin;%PATH%
set INCLUDE=%VCINSTALLDIR%\INCLUDE;%VCINSTALLDIR%\MFC\INCLUDE;%VCINSTALLDIR%\ATL\INCLUDE;%INCLUDE%
set LIB=%VCINSTALLDIR%\LIB;%VCINSTALLDIR%\MFC\LIB;%LIB%
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=cl
set COMPFLAGS=-c -Zp8 -G5 -W3 -EHs -DMATLAB_MEX_FILE -nologo /MD
set OPTIMFLAGS=-O2 -Oy- -DNDEBUG
set DEBUGFLAGS=-Zi -Fd"%OUTDIR%%MEX_NAME%.pdb"
set NAME_OBJECT=/Fo

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\microsoft\
set LINKER=link
set LINKFLAGS=/dll /export:%ENTRYPOINT% /MAP /LIBPATH:"%LIBLOC%" libmx.lib libmex.lib libmat.lib /implib:"%LIB_NAME%.x" kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /MACHINE:IX86
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/debug
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT=/out:"%OUTDIR%%MEX_NAME%%MEX_EXT%"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=rc /fo "%OUTDIR%mexversion.res"
set RC_LINKER=

set POSTLINK_CMDS=del "%OUTDIR%%MEX_NAME%.map"
set POSTLINK_CMDS1=del "%LIB_NAME%.x"
