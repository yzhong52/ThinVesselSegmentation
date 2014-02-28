#define _CRT_SECURE_NO_DEPRECATE
#include "stdafx.h"
#include <time.h>
#include "VesselDetector.h" // For computing vesselness
#include "GLViwerWrapper.h" // For visualization

GLViewerExt viewer;

namespace sample_code{
	int vesselness(void);
}

int main(int argc, char* argv[])
{
	sample_code::vesselness();
	return 0;
}


namespace sample_code{
	int vesselness(void){
		// create output folders if it does not exist
		CreateDirectory(L"./output", NULL);

		bool flag = false;

		// Original data name
		string dataname = "data15";
		// Parameters for Vesselness
		float sigma_from = 0.5f;
		float sigma_to = 8.0f;
		float sigma_step = 1.0f;
		// Parameters for vesselness
		float alpha = 1.0e-1f;	
		float beta  = 5.0e0f;
		float gamma = 3.5e5f;

		
		// laoding data
		Data3D<short> im_short;
		bool falg = im_short.load( "../data/" + dataname+".data" );
		if(!falg) return 0;
		
		Data3D<Vesselness_Nor> vn_nor, vn_nor2;
		VesselDetector::hessien( im_short, vn_nor, 0, 1.0f, alpha, beta, gamma ); // TODO: remove this function
		VesselDetector::hessien2( im_short, vn_nor2, 0, 1.0f, alpha, beta, gamma );

		// Visualize result with maximum intensity projection (MIP)
		viewer.addObject( im_short, GLViewer::Volumn::MIP );
		viewer.addObject( vn_nor,   GLViewer::Volumn::MIP );
		viewer.addObject( vn_nor2,  GLViewer::Volumn::MIP );
		viewer.go(600, 200, 3);
		return 0;
	}
}