#pragma once

#include "opencv2\core\core.hpp"
using namespace cv; 

class Line3D; 
template <typename T> class Data3D; 
template <typename T> class ModelSet; 

extern const double LOGLIKELIHOOD;
extern const double PAIRWISESMOOTH; 

class LevenburgMaquart
{
public:
	void reestimate(const vector<Vec3i>& dataPoints,
		const vector<int>& labelings, 
		const ModelSet<Line3D>& modelset, 
		const Data3D<int>& indeces ); 
};

