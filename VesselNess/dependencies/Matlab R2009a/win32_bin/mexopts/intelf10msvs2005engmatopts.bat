@echo off  
rem INTEL10MSVS2005ENGMATOPTS.BAT
rem
rem    Compile and link options used for building stand-alone engine or 
rem    MAT programs with the Intel® Visual Fortran Compiler 10.1 with the
rem    Microsoft® Visual Studio® 2005 linker.
rem    
rem    $Revision: 1.1.6.2 $  $Date: 2008/01/10 20:49:55 $
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
set COMPFLAGS=/fpp /Qprec /I"%MATLAB%/extern/include" /c /nologo /fixed /fp:source /MD /assume:bscc
set OPTIMFLAGS=/Ox /DNDEBUG
set DEBUGFLAGS=/Zi /Fd"%OUTDIR%%MEX_NAME%.pdb"
set NAME_OBJECT=/Fo

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\microsoft
set LINKER=link
set LINKFLAGS=/LIBPATH:"%LIBLOC%" libmx.lib libmat.lib libeng.lib /nologo /subsystem:console
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/debug /PDB:"%OUTDIR%%MEX_NAME%.pdb" /INCREMENTAL:NO
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT=/out:"%OUTDIR%%MEX_NAME%.exe"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=
set RC_LINKER= 
set POSTLINK_CMDS1=mt -outputresource:"%OUTDIR%%MEX_NAME%.exe";1 -manifest "%OUTDIR%%MEX_NAME%.exe.manifest" 
set POSTLINK_CMDS2=del "%OUTDIR%%MEX_NAME%.exe.manifest" 