Model Fitting (Levenberg Marquardt)
========================

Using Levenberg Marquardt algorithm for energy minimization. 

Energy contains two parts:

- Data cost
  Distance to the center of a line model
   
- Pair-wise smooth cost
  Complicated. Please refer to this [paper](http://www.csd.uwo.ca/~yuri/Abstracts/cvpr12-abs.shtml) for more details

Levenberg Marquardt algorithm requires to compute the Jacobin matrix for both the data cost and the smooth cost. This computation is very time-consuming and the computation has been highly paralleled in this version. 
