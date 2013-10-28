@echo off
rem INTELC91MSVS2005OPTS.BAT
rem
rem    Compile and link options used for building MEX-files using the
rem    Intel® C++ 9.1 compiler with the Microsoft® Visual Studio® 2005
rem    linker
rem
rem StorageVersion: 1.0
rem C++keyFileName: INTELC91MSVS2005OPTS.BAT
rem C++keyName: Intel C++
rem C++keyManufacturer: Intel
rem C++keyVersion: 9.1
rem C++keyLanguage: C++
rem
rem    $Revision: 1.1.6.6 $  $Date: 2008/05/27 18:20:58 $
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************
set MATLAB=%MATLAB%
set ICPP_COMPILER91=%ICPP_COMPILER91%
set VS80COMNTOOLS=%VS80COMNTOOLS%
set LINKERDIR=%VS80COMNTOOLS%\..\..
set PATH=%ICPP_COMPILER91%\Ia32\Bin;%LINKERDIR%\VC\BIN;%LINKERDIR%\Common7\Tools;%LINKERDIR%\Common7\Tools\bin;%LINKERDIR%\Common7\IDE;%LINKERDIR%\SDK\v2.0\bin;%PATH%
set INCLUDE=%ICPP_COMPILER91%\Ia32\Include;%LINKERDIR%\VC\ATLMFC\INCLUDE;%LINKERDIR%\VC\INCLUDE;%LINKERDIR%\VC\PlatformSDK\include;%LINKERDIR%\SDK\v2.0\include;%INCLUDE%
set LIB=%ICPP_COMPILER91%\Ia32\Lib;%LINKERDIR%\VC\ATLMFC\LIB;%LINKERDIR%\VC\LIB;%LINKERDIR%\VC\PlatformSDK\lib;%LINKERDIR%\SDK\v2.0\lib;%MATLAB%\extern\lib\win32;%LIB%
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=icl
set COMPFLAGS=-c -Zp8 -G5 -W3 -EHs -DMATLAB_MEX_FILE -nologo /MD
set OPTIMFLAGS=-O2 -Oy- -DNDEBUG
set DEBUGFLAGS=/Z7
set NAME_OBJECT=/Fo

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
set LIBLOC=%MATLAB%\extern\lib\win32\microsoft
set LINKER=link
set LINKFLAGS=/dll /export:%ENTRYPOINT% /LIBPATH:"%LIBLOC%" libmx.lib libmex.lib libmat.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /implib:"%LIB_NAME%.x" /MAP:"%OUTDIR%%MEX_NAME%%MEX_EXT%.map" /NOLOGO /INCREMENTAL:NO
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/debug /pdb:"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb"
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
