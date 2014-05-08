Thin Vessels Segmentation
========================

Vesselness
---------------
Compute vesselness measure using [Frangi](http://link.springer.com/chapter/10.1007/BFb0056195#page-1)'s method. 
Please refer to `/Thin_Vessel_Segmentation/VesselNess` for more details. 

ModelFitting
---------------
Fitting geometric models (lines) to the 3D vessel data. 
Please refer to `/Thin_Vessel_Segmentation/ModelFitting` for more details. 

Others
---------------
The following are important components using in the projects:

-  SparseMatrix: Sparse matrix representation, contains all necessary matrix manipulations
-  SparseMatrixCV: A sparsematrix wrapper for OpenCV
-  Vesselness-cuda: cuda version of the vesselness measure, still under construction
-  EigenDecomp: eigenvalue decomposition of a 3 by 3 symetric matrix


Requirements
---------------
1. A graphic card which supports OpenGL 4.0
2. Microsoft Visual Studio 2010 (or higher) 

3. This project requires freeglut 2.8.1, glew 1.10.0 and OpenCV 2.8.4. You have to add the following directories to PATH: 

 - `%CD%\dependencies\freeglut 2.8.1\x64;`
 - `%CD%\dependencies\glew-1.10.0\bin\x64;`
 - `%CD%\dependencies\OpenCV 2.4.3\x64-vc10;`

Note: `%CD%` above means the location where you save the files. 

Or you can copy the dlls in their bin folders to the project directory either manually or excute the bat file `set-up-dlls.bat`. 
