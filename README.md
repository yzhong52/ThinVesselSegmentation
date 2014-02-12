This is a cpp implementation of Vesselness Measure for 3D volume based on the following paper. 

Frangi, Alejandro F., et al. "Multiscale vessel enhancement filtering." Medical Image Computing and Computer-Assisted Interventation—MICCAI’98. Springer Berlin Heidelberg, 1998. 130-137.


Excutable Files are provided
========================
Excutable files and example 3D volume are provided.
The data and the result are visulized with maximum intensity projection. 

Requirements
========================
 - A graphic card which supports OpenGL 4.0
 - Microsoft Visual Studio 2010 (or higher) 

Before Compile
========================

This project requires freeglut, glew and OpenCV. You do not have to install these packages because they are included in the project. But you do need to add the following to path: 

%CD%\dependencies\freeglut 2.8.1\lib\x64;

%CD%\dependencies\glew-1.10.0\bin\Release\x64;

%CD%\dependencies\OpenCV 2.4.3\x64\vc10;

Note: %CD% above means the location where you save the files. 

