#define _CRT_SECURE_NO_DEPRECATE
#include "stdafx.h"
#include <time.h>
#include "VesselDetector.h" // For computing vesselness
#include "GLViwerWrapper.h" // For visualization

GLViewerExt viewer;

namespace sample_code{
	int vesselness( string dataname = "data15" );
	int centreline( string dataname = "data15" ); 
}

int main(int argc, char* argv[])
{
	// sample_code::vesselness();
	sample_code::centreline(); 
	return 0;
}


namespace sample_code{
	int vesselness( string dataname ){
		// create output folders if it does not exist
		CreateDirectory(L"../temp", NULL);

		bool flag = false;

		// Parameters for Vesselness
		float sigma_from = 1.0f;
		float sigma_to = 8.10f;
		float sigma_step = 0.5f;
		// Parameters for vesselness
		float alpha = 1.0e-1f;	
		float beta  = 5.0e0f;
		float gamma = 3.5e5f;

		// laoding data
		Data3D<short> im_short;
		bool falg = im_short.load( "../data/" + dataname + ".data" );
		
		if(!falg) return 0;
		
		Data3D<Vesselness_Sig> vn_sig; 
		VesselDetector::compute_vesselness( im_short, vn_sig, 
			sigma_from, sigma_to, sigma_step,
			alpha, beta, gamma );
		vn_sig.save( "../temp/" + dataname + ".vn_sig" );
		viewer.addObject( vn_sig,  GLViewer::Volumn::MIP );
		viewer.addDiretionObject( vn_sig );

		viewer.go(600, 400, 2);

		return 0;
	}


	int centreline( string dataname ) {
		Data3D<Vesselness_Sig> vn_sig; 
		Data3D<Vesselness_Sig> vn_sig_nms; 
		vn_sig.load( "../temp/" + dataname + ".vn_sig" );
		IP::non_max_suppress( vn_sig, vn_sig_nms );

		viewer.addObject( vn_sig,  GLViewer::Volumn::MIP );
		viewer.addDiretionObject( vn_sig );
		viewer.addObject( vn_sig_nms,  GLViewer::Volumn::MIP );
		viewer.addDiretionObject( vn_sig_nms );
		
		viewer.go(600, 400, 4);
		return 0;
	}
}


