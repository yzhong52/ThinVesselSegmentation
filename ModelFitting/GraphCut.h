#pragma once

#include "opencv2/core/core.hpp"
using namespace cv; 

class Line3D; 
extern const double LOGLIKELIHOOD;

class GraphCut
{
public:
	static double estimation( const vector<Vec3i>& dataPoints,
		vector<int>& labelings, 
		const vector<Line3D*>& lines );
};

