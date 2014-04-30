Model Fitting
========================

Using Levenberg Marquart algorithm for energy minimization. 

Enery contains two parts:

- Data cost
  Distance to the center of a line model
   
- Pair-wise smooth cost
  Complicated. Please refer to this [paper](http://www.csd.uwo.ca/~yuri/Abstracts/cvpr12-abs.shtml) for more details

Levenberg Marquart algorithm requires to compute the Jacobian matrix for both the data cost and the smooth cost. This computation is very time-consuming and the computation has been higly parallelized. 

Major Updates in current version
========================
1) Parallelized computating the Jacobain matrix of both data cost and smooth cost
2) Profiling with Vtune

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

