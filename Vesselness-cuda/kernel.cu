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

// files created for this project
#include "ImageProcessing.cuh" 
#include "VesselnessFilter.cuh"



int main()
{
	Data3D<short> im_short;
	im_short.load( "../data/data15.data" );
	
	Data3D<int> im_blurred;
	IPG::GaussianBlur3D( im_short, im_blurred, 5, 1.0 );

	// Computer Vesselness Measure
	Data3D<float> im_float;
	VFG::compute_vesselness( im_short, im_float, 1.0f, 2.0f, 0.5f );
	

	// Visualize result with maximum intensity projection (MIP)
	viewer.addObject( im_blurred, GLViewer::Volumn::MIP );
	viewer.addObject( im_float, GLViewer::Volumn::MIP );
	viewer.go(400, 200, 2);

    return 0;
}