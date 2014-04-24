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
		CreateDirectory(L"../temp", NULL);

		bool flag = false;

		// Original data name
		string dataname = "data15";
		// Parameters for Vesselness
		float sigma_from = 1.0f;
		float sigma_to = 8.0f;
		float sigma_step = 0.5f;
		// Parameters for vesselness
		float alpha = 1.0e-1f;	
		float beta  = 5.0e0f;
		float gamma = 3.5e5f;

		// laoding data
		Data3D<short> im_short;
		bool falg = im_short.load( "../data/" + dataname + ".data" );
		if(!falg) return 0;
		
		// Two different ways of computing Vesselness measure
		
		// 1. old, slower 
		 Data3D<Vesselness_All> vn_all0;
		 VesselDetector::compute_vesselness( im_short, vn_all0, 
			sigma_from, sigma_to, sigma_step,
			alpha, beta, gamma );
		 viewer.addObject( vn_all0, GLViewer::Volumn::MIP );

		// 2. new, faster 
		Data3D<Vesselness_All> vn_all; 
		VesselDetector::compute_vesselness2( im_short, vn_all, 
			sigma_from, sigma_to, sigma_step,
			alpha, beta, gamma );
		viewer.addObject( vn_all,  GLViewer::Volumn::MIP );
		Data3D<Vesselness_Sig> vn_sig( vn_all );
		vn_sig.save( "../temp/roi15.vn_sig" ); 
		
		// Visualize result with maximum intensity projection (MIP)
		viewer.addObject( im_short, GLViewer::Volumn::MIP );

		// visualize the data in thre viewports
		viewer.go(600, 200, 3);

		return 0;
	}
}