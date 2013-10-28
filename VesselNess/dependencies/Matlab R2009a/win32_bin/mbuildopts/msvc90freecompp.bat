@echo off
rem MSVC90COMPP.BAT
rem
rem    Compile and link options for use with MATLAB Compiler.
rem    using the Microsoft Visual Studio 2008 Express Edition compiler.
rem
rem    $Revision: 1.1.8.1 $  $Date: 2008/11/04 19:40:22 $
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************
set MATLAB=%MATLAB%
set VSINSTALLDIR=%VS90COMNTOOLS%\..\..
set VCINSTALLDIR=%VSINSTALLDIR%\VC
rem In this case, LINKERDIR is being used to specify the location of the SDK
set LINKERDIR='.registry_lookup("SOFTWARE\Microsoft\Microsoft SDKs\Windows" , "CurrentInstallFolder").'
set PATH=%VCINSTALLDIR%\BIN\;%VSINSTALLDIR%\VC\bin;%LINKERDIR%\bin;%VSINSTALLDIR%\Common7\IDE;%VSINSTALLDIR%\Common7\Tools;%VSINSTALLDIR%\Common7\Tools\bin;%VCINSTALLDIR%\VCPackages;%MATLAB_BIN%;%PATH%
set INCLUDE=%VCINSTALLDIR%\ATLMFC\INCLUDE;%VCINSTALLDIR%\INCLUDE;%LINKERDIR%\include;%INCLUDE%
set LIB=%VCINSTALLDIR%\ATLMFC\LIB;%VCINSTALLDIR%\LIB;%LINKERDIR%\lib;%VSINSTALLDIR%\SDK\v2.0\lib;%MATLAB%\extern\lib\win32;%LIB%

set PERL="%MATLAB%\sys\perl\win32\bin\perl.exe"   
set MW_TARGET_ARCH=win32

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=cl
set OPTIMFLAGS=-O2 -DNDEBUG
set DEBUGFLAGS=-Z7
set VER_SPECIFIC_OPTS=/D_CRT_SECURE_NO_DEPRECATE
set CPPOPTIMFLAGS=-O2 -DNDEBUG
set CPPDEBUGFLAGS=-Zi -Fd"%OUTDIR%%MEX_NAME%.pdb"
set COMPFLAGS=-c -Zp8 -GR -W3 -EHsc- -Zc:wchar_t- -nologo %VER_SPECIFIC_OPTS%
set CPPCOMPFLAGS=-c -Zp8 -GR  -W3 -EHsc- -Zc:wchar_t- -nologo -Zm500 -MD -I"%MATLAB%\extern\include\cpp" -DMSVC -DIBMPC /D_SECURE_SCL=0 %VER_SPECIFIC_OPTS%
set DLLCOMPFLAGS=-c -Zp8 -GR -EHsc- -Zc:wchar_t- -W3 -nologo -I"%MATLAB%\extern\include\win32" -DMSVC -DIBMPC %VER_SPECIFIC_OPTS%
rem set OPTIMFLAGS=/MD -O2 -Oy- -DNDEBUG
rem set DEBUGFLAGS=/MD -Zi -Fd"%OUTDIR%%MEX_NAME%%MEX_EXT%.pdb"
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
set LINKFLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib oleaut32.lib ole32.lib uuid.lib odbc32.lib odbccp32.lib /LIBPATH:"%LIBLOC%" /nologo 
set LINKFLAGS=%LINKFLAGS% mclmcrrt.lib %MCR_DELAYLIB% %MCR_DELAYLOAD%
set CPPLINKFLAGS=
set DLLLINKFLAGS= %LINKFLAGS% /dll /implib:"%OUTDIR%%MEX_NAME%.lib" /def:%DEF_FILE%
set HGLINKFLAGS=
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=/debug /PDB:"%OUTDIR%%MEX_NAME%.pdb"
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
