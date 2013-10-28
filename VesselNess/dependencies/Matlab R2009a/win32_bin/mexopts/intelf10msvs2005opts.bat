@echo off
rem INTELF10MSVS2005OPTS.BAT
rem
rem    Compile and link options used for building MEX-files using the
rem    Intel® Fortran Compiler 10.1 with the Microsoft® Visual Studio®
rem    2005 linker.
rem
rem StorageVersion: 1.0
rem FortrankeyFileName: INTELF10MSVS2005OPTS.BAT
rem FortrankeyName: Intel Visual Fortran
rem FortrankeyManufacturer: Intel
rem FortrankeyVersion: 10.1
rem FortrankeyLanguage: Fortran
rem
rem    $Revision: 1.1.6.3 $  $Date: 2008/05/27 18:20:59 $
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************
set MATLAB=%MATLAB%
set IFORT_COMPILER10=%IFORT_COMPILER10%
set VS80COMNTOOLS=%VS80COMNTOOLS%
set LINKERDIR=%VS80COMNTOOLS%\..\..
set PATH=%IFORT_COMPILER10%\IA32\Bin;%LINKERDIR%\VC\BIN;%LINKERDIR%\Common7\Tools;%LINKERDIR%\Common7\Tools\bin;%LINKERDIR%\Common7\IDE;%LINKERDIR%\SDK\v2.0\bin;%PATH%
set INCLUDE=%IFORT_COMPILER10%\IA32\Include;%LINKERDIR%\VC\ATLMFC\INCLUDE;%LINKERDIR%\VC\INCLUDE;%LINKERDIR%\VC\PlatformSDK\include;%LINKERDIR%\SDK\v2.0\include;%INCLUDE%
set LIB=%IFORT_COMPILER10%\IA32\Lib;%LINKERDIR%\VC\ATLMFC\LIB;%LINKERDIR%\VC\LIB;%LINKERDIR%\VC\PlatformSDK\lib;%LINKERDIR%\SDK\v2.0\lib;%MATLAB%\extern\lib\win32;%LIB%
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=ifort
set COMPFLAGS=/fpp /Qprec "/I%MATLAB%/extern/include" -c -nologo -DMATLAB_MEX_FILE /fixed /MD /fp:source /assume:bscc
set OPTIMFLAGS=-Ox -DNDEBUG
set DEBUGFLAGS=/Z7
set NAME_OBJECT=/Fo

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\microsoft
set LINKER=link
set LINKFLAGS=/DLL /EXPORT:MEXFUNCTION /LIBPATH:"%LIBLOC%" libmx.lib libmex.lib libmat.lib /implib:"%LIB_NAME%.x" /MAP:"%OUTDIR%%MEX_NAME%%MEX_EXT%.map" /NOLOGO /INCREMENTAL:NO
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/debug /PDB:"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb"
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT=/out:"%OUTDIR%%MEX_NAME%%MEX_EXT%"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=rc /fo "%OUTDIR%mexversion.res"
set RC_LINKER=

set POSTLINK_CMDS=del "%OUTDIR%%MEX_NAME%%MEX_EXT%.map"
set POSTLINK_CMDS1=del "%LIB_NAME%.x" "%LIB_NAME%.exp"
set POSTLINK_CMDS2=mt -outputresource:"%OUTDIR%%MEX_NAME%%MEX_EXT%";2 -manifest "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest"
set POSTLINK_CMDS3=del "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest" 
