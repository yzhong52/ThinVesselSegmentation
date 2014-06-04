Thin Vessels Segmentation
========================

Vesselness
---------------
Compute vesselness measure using [Frangi](http://link.springer.com/chapter/10.1007/BFb0056195#page-1)'s method. 
Please refer to `/Thin_Vessel_Segmentation/VesselNess` for more details. 

ModelFitting
---------------
Fitting geometric models (lines) to the 3D vessel data using Levenberg-Marquardt algorithm. 
Please refer to `/Thin_Vessel_Segmentation/ModelFitting` for more details. 

Others
---------------
The following are important components using in the projects:

-  SparseMatrix: Sparse matrix representation, contains all necessary matrix manipulations
-  SparseMatrixCV: A wrapper of SparseMatrix for OpenCV
-  Vesselness-cuda: CUDA version of the vesselness measure (under construction)
-  RingsReduction: Rings reduction of CT image (under developing)
-  EigenDecomp: eigenvalue decomposition of a 3 by 3 symmetric matrix




Requirements (Linux)
---------------

1. Install `freeglut`

	`sudo apt-get install freeglut3-dev`

2. Install the X Window System (X11, X, and sometimes informally X-Windows), which is a windowing system for bitmap displays

	`sudo apt-get install libxmu-dev`

	`sudo apt-get install libxi-dev`

3. Install glew

	`sudo apt-get install libglew-dev`

4. Install OpenGL

	`sudo apt-get install mesa-common-dev`

5. Install OpenCV 2.4.9

	Download [OpenCV](http://opencv.org/). Generate makefile witl cmake, build and install the liabrary. 

	`make`

	`make install`