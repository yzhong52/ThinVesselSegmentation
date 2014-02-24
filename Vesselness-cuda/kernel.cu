// Header files for opencv
#include <iostream> 
#include <time.h>
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
#include "VesselDetector.h"


int main()
{
	clock_t t;

	Data3D<short> im_short;
	bool flag = im_short.load( "../data/data15.data" );
	if(!flag) return 0; 
	
	Data3D<int> im_blurred;
	IPG::GaussianBlur3D( im_short, im_blurred, 5, 1.0 );

	// Computer Vesselness Measure
	Data3D<float> im_float;
	t = clock();
	VFG::compute_vesselness( im_short, im_float, 0.7f, 1.6f, 0.2f );
	t = clock() - t;
	printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	
	Data3D<Vesselness_All> im_float2;
	t = clock(); 
	VF::compute_vesselness( im_short, im_float2, 0.7f, 1.6f, 0.2f );
	t = clock() - t; 
	printf ("It took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	
	// Visualize result with maximum intensity projection (MIP)
	im_short.remove_margin( 7 ); 
	im_float.remove_margin( 7 );
	im_float2.remove_margin( 7 ); 
	viewer.addObject( im_short, GLViewer::Volumn::MIP );
	viewer.addObject( im_float, GLViewer::Volumn::MIP );
	viewer.addObject( im_float2, GLViewer::Volumn::MIP );
	viewer.go(600, 200, 3);





    return 0;
}