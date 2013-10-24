// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#define _CRT_SECURE_NO_DEPRECATE

#include <iostream>
#include <vector>

#include <opencv\cv.h>
#include <opencv2\opencv.hpp>
//#include <opencv2\contrib\contrib.hpp>

#include "smart_assert.h"

using namespace std;
using namespace cv;

// Get OpenCV type from type_info
// Example: 
//     CV_TYPE(typeid(short)) returns CV_16S
//     CV_TYPE(typeid(int))   returns CV_32S
int CV_TYPE(const type_info& type);

// Get String Descriptor from type_info
string STR_TYPE(const type_info& type);

static const short MAX_SHORT = unsigned short(1<<15)-1;
static const short MIN_SHORT = short(1<<15);


