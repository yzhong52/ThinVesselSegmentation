Thin Vessels Segmentation
========================

Vesselness
---------------
Compute vesselness measure using [Frangi](http://link.springer.com/chapter/10.1007/BFb0056195#page-1)'s method. Please refer to `/Thin_Vessel_Segmentation/VesselNess` for more details. 

ModelFitting
---------------
Fitting geometric models (lines) to the 3D vessel data. Please refer to `/Thin_Vessel_Segmentation/ModelFitting` for more details. 
Others
---------------
The following are important components using in the projects:

-  SparseMatrix: Sparse matrix representation, contains all necessary matrix manipulations
-  SparseMatrixCV: A sparsematrix wrapper for OpenCV
-  Vesselness-cuda: cuda version of the vesselness measure, still under construction
-  EigenDecomp: eigenvalue decomposition of a 3 by 3 symetric matrix
