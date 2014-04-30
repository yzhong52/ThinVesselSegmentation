Model Fitting
========================

Using Levenburg Maquart algorithm for energy minimization. 

Enery contains two parts:

1) data cost
   Distance to the center of a line model
2) Pair-wise Smooth Cost
   Complicated. Please refer to this [paper](http://www.csd.uwo.ca/~yuri/Abstracts/cvpr12-abs.shtml)

Requirements
========================
 - A graphic card which supports OpenGL 4.0
 - Microsoft Visual Studio 2010 (or higher) 

Before Compile
========================

This project requires freeglut 2.8.1, glew 1.10.0 and OpenCV 2.8.4. You have to add the following directories to PATH: 

%CD%\dependencies\freeglut 2.8.1\x64;

%CD%\dependencies\glew-1.10.0\bin\x64;

%CD%\dependencies\OpenCV 2.4.3\x64-vc10;

Note: %CD% above means the location where you save the files. 

