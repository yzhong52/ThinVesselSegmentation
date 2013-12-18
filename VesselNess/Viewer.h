#pragma once

#define _CRT_SECURE_NO_DEPRECATE

#include "stdafx.h"

namespace Viewer
{
	namespace OpenCV {
		void plot( Mat_<double> mx, Mat_<double> my );
		void plot( const string& name,    // name of the output image
			vector<Mat_<double>>& mat_ys, // data y
			int im_height = 200,          // image height
			int im_width = 0,             // image width, will be computed based on the size of mat_ys if left empty
			Mat_<unsigned char> mat_bg = Mat_<unsigned char>());
	}
};

namespace Visualizer = Viewer;
namespace VI = Viewer;

