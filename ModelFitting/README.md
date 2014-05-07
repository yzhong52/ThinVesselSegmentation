Model Fitting
========================

Using Levenberg Marquart algorithm for energy minimization. 

Enery contains two parts:

- Data cost
  Distance to the center of a line model
   
- Pair-wise smooth cost
  Complicated. Please refer to this [paper](http://www.csd.uwo.ca/~yuri/Abstracts/cvpr12-abs.shtml) for more details

Levenberg Marquart algorithm requires to compute the Jacobian matrix for both the data cost and the smooth cost. This computation is very time-consuming and the computation has been higly parallelized in this version. 
