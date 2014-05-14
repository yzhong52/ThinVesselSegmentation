#define _CRT_SECURE_NO_DEPRECATE
#include "stdafx.h"
#include <time.h>
#include "VesselDetector.h" // For computing vesselness
#include "GLViwerWrapper.h" // For visualization
#include "../ModelFitting/SyntheticData.h"

GLViewerExt viewer;

#define INPUT_DIR "../data/"
#define OUTPUT_DIR "../temp/"

namespace sample_code{
	// Compute vesselness measure
	int vesselness( bool isDisplay, string dataname = "data15" );

	// Extract Vessel centrelines with non-maximum suppression 
	int centreline( bool isDisplay, string dataname = "data15" ); 
}

namespace experiment{
	void synthetic_yes(); 
}


int main(int argc, char* argv[])
{
	experiment::synthetic_yes(); 
	//sample_code::vesselness( false, "data15" );
	//sample_code::centreline( false, "data15" ); 
	return 0;
}


namespace sample_code{
	int vesselness( bool isDisplay, string dataname ){
		// create output folders if it does not exist
		CreateDirectory(L"../temp", NULL);

		// Sigma: Parameters for Vesselness
		// [sigma_from, sigma_to]: the potential size rang of the vessels
		// sigma_step: precision of computation
		float sigma_from = 1.0f;
		float sigma_to = 8.10f;
		float sigma_step = 0.3f;
		// Parameters for vesselness, please refer to Frangi's papaer 
		// or this [blog](http://yzhong.co/?p=351)
		float alpha = 1.0e-1f;	
		float beta  = 5.0e0f;
		float gamma = 3.5e5f;

		// laoding data
		Data3D<short> im_short;
		bool falg = im_short.load( INPUT_DIR + dataname + ".data" );
		
		// Compute Vesselness
		Data3D<Vesselness_Sig> vn_sig; 
		VesselDetector::compute_vesselness( im_short, vn_sig, 
			sigma_from, sigma_to, sigma_step,
			alpha, beta, gamma );
		vn_sig.save( OUTPUT_DIR + dataname + ".vn_sig" );

		// If you want to visulize the data using Maximum-Intensity Projection
		if( isDisplay ) {
			viewer.addObject( vn_sig,  GLViewer::Volumn::MIP );
			viewer.addDiretionObject( vn_sig );
			viewer.go(600, 400, 2);
		}

		return 0;
	}


	int centreline( bool isDisplay, string dataname ) {
		Data3D<Vesselness_Sig> vn_sig; 
		Data3D<Vesselness_Sig> vn_sig_nms; 
		vn_sig.load( OUTPUT_DIR + dataname + ".vn_sig" );
		IP::non_max_suppress( vn_sig, vn_sig_nms );
		vn_sig_nms.save( OUTPUT_DIR + dataname + ".nms.vn_sig" );

		if( isDisplay ) {
			viewer.addObject( vn_sig,  GLViewer::Volumn::MIP );
			viewer.addDiretionObject( vn_sig );
			viewer.addObject( vn_sig_nms,  GLViewer::Volumn::MIP );
			viewer.addDiretionObject( vn_sig_nms );
		
			viewer.go(600, 400, 4);
		}
		return 0;
	}
}




void experiment::synthetic_yes(){
	Data3D<short> im_short; 
	SyntheticData::Yes( im_short ); 
	im_short.save( "../temp/yes.data" ); 
	sample_code::vesselness( false, "../temp/yes" );
	sample_code::centreline( true, "../temp/yes" ); 
}