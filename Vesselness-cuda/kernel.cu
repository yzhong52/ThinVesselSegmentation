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
#include "VesselnessFilterPartial.cuh"
#include "VesselDetector.h"



int main()
{
	clock_t t;

	bool flag; 
	Data3D<short> im_short;
	//flag = im_short.load( "../temp/vessel3d.rd.19.442.data" );
	flag = im_short.load( "../temp/vessel3d.mmrd.17data" ); 
	//flag = im_short.load( "../temp/roi20.data" ); 
	//flag = im_short.load( "../temp/data/roi16.partial.data" ); 
	if(!flag) return 0; 
	
	// Compute Vesselness Measure
	Data3D<float> im_float;
	t = clock();
	VFG::compute_vesselness_partial( im_short, im_float, 
		1.0f, 4.1f, 0.5f,
		1.0e-1f, 5.0f, 3.5e5f,
		300,300,300);
	t = clock() - t;
	cout << "It took me " << t << " clicks (" << float(t)/CLOCKS_PER_SEC << " seconds). " << endl; 
	
	// Visualize result with maximum intensity projection (MIP)
	viewer.addObject( im_short, GLViewer::Volumn::MIP );
	viewer.addObject( im_float, GLViewer::Volumn::MIP );
	
	viewer.go(600, 300, 2);

    return 0;
}
