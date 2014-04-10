// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#ifndef _CRT_SECURE_NO_DEPRECATE
	#define _CRT_SECURE_NO_DEPRECATE
#endif


#include <vector>
#include <iostream> 
using namespace std;
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv; 

#include "smart_assert.h"

using namespace std;
using namespace cv;

//// Get OpenCV type from type_info
//// Example: 
////     CV_TYPE(typeid(short)) returns CV_16S
////     CV_TYPE(typeid(int))   returns CV_32S
//int CV_TYPE(const type_info& type);
//
//// Get String Descriptor from type_info
//string STR_TYPE(const type_info& type);


