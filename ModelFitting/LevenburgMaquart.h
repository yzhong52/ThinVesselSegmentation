#pragma once

#include "opencv2/core/core.hpp"
using namespace cv; 

class Line3D; 

class LevenburgMaquart
{
public:
	LevenburgMaquart(void);
	~LevenburgMaquart(void);
	static void reestimate(const vector<Vec3i>& dataPoints,
		const vector<int>& labelings, 
		const vector<Line3D*>& lines );
};

