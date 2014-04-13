#pragma once

#include "opencv2\core\core.hpp"
using namespace cv; 

class Line3D; 
template <typename T> class Data3D; 

extern const double LOGLIKELIHOOD;
extern const double PAIRWISESMOOTH; 

class LevenburgMaquart
{
public:
	void reestimate(const vector<Vec3i>& dataPoints,
		const vector<int>& labelings, 
		const vector<Line3D*>& lines, 
		const Data3D<int>& indeces );
};

