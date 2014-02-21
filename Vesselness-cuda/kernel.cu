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
#include "VesselDetector.h"


int main()
{
	Data3D<short> im_short;
	//im_short.load( "../data/data15.data" );
	im_short.reset( Vec3i(30,30,30) );
	for( int i=0; i<30; i++ ) im_short.at(i, 13, 13) = 500; 
	
	Data3D<int> im_blurred;
	IPG::GaussianBlur3D( im_short, im_blurred, 5, 1.0 );

	// Computer Vesselness Measure
	Data3D<float> im_float;
	VFG::compute_vesselness( im_short, im_float, 1.5f, 1.6f, 0.5f );
	

	Data3D<Vesselness_All> im_float2;
	VesselDetector::compute_vesselness( im_short, im_float2, 1.5f, 1.6f, 0.5f );
	




	for( int x=3; x<8; x++ ) for( int y=3; y<8; y++ ) {
		cout << x << "," << y << "," << 5 << "\t";
		cout << im_float.at( x,y,5) << "\t";
		cout << im_float2.at(x,y,5).rsp << endl;
	}
	cout << endl;
	

	
		


	// Visualize result with maximum intensity projection (MIP)
	viewer.addObject( im_short, GLViewer::Volumn::MIP );
	viewer.addObject( im_float, GLViewer::Volumn::MIP );
	viewer.addObject( im_float2, GLViewer::Volumn::MIP );
	viewer.go(600, 200, 3);

    return 0;
}