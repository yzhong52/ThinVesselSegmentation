// Header files for opencv
#include <iostream> 
using namespace std;
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv; 

#include "VesselDetector.h"
#include "Data3D.h"
#include "nstdio.h"
#include "VesselNess.h"
#include "TypeInfo.h"
#include "GLViwerWrapper.h" // For visualization



GLViewerExt viewer;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// files created for this project
#include "ImageProcessing.cuh" 


int main()
{
	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };
	//
	//IPG::addWithCuda( c, a, b, 5 ); 

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);



	Data3D<short> im_short;
	im_short.load( "../data/data15.data" );
	Data3D<float> im_int;
	im_short.convertTo( im_int ); 
	
	Data3D<int> im_blurred;
	IPG::GaussianBlur3D( im_int, im_blurred, 5, 1.0 );

	//// Visualize result with maximum intensity projection (MIP)
	viewer.addObject( im_blurred, GLViewer::Volumn::MIP );
	viewer.addObject( im_short, GLViewer::Volumn::MIP );
	viewer.go(400, 200, 2);

    return 0;
}