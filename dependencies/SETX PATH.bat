
SET NEW_USER_PATH=%CD%\freeglut 2.8.1\x64;%NEW_USER_PATH%
:: This is a comment
:: Uncomment the following line for 32-bit machine
:: SET NEW_USER_PATH=%CD%\freeglut 2.8.1\lib\x86;%NEW_USER_PATH%

SET NEW_USER_PATH=%CD%\glew-1.10.0\x64;%NEW_USER_PATH%
:: This is a comment
:: Uncomment the following line for 32-bit machine
:: SET NEW_USER_PATH=%CD%\glew-1.10.0\bin\Release\Win32;%NEW_USER_PATH%

SET NEW_USER_PATH=%CD%\opencv2.4.8\x64-mingw32-w64;%NEW_USER_PATH%
SET NEW_USER_PATH=%CD%\opencv2.4.8\x64-vc10;%NEW_USER_PATH%
:: This is a comment
:: Uncomment the following line for 32-bit machine
:: SET NEW_USER_PATH=%CD%\OpenCV 2.4.8\x86\vc10\bin;%NEW_USER_PATH%

setx PATH "%NEW_USER_PATH%"

PAUSE
