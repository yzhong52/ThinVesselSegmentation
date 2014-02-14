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
	
		// vesselness output file name
		stringstream vesselness_name;
		vesselness_name << "output/";
		vesselness_name << dataname;
		vesselness_name << ".sigma_to" << sigma_to;

		// logging information
		stringstream vesselness_log;
		vesselness_log << "alpha = " << alpha << endl;
		vesselness_log << "beta = " << beta << endl;
		vesselness_log << "gamma = " << gamma << endl;
		vesselness_log << "Vesselness is computed with the following sigmas: ";
		for( float sigma = sigma_from; sigma < sigma_to; sigma += sigma_step ) {
			vesselness_log << sigma << ","; 
		}
		vesselness_log << '\b' << endl; // remove the last ','

		clock_t start = clock();
		// compute vesselness
		Data3D<Vesselness_All> vn_all;
		VD::compute_vesselness( im_short, vn_all, sigma_from, sigma_to, sigma_step, alpha, beta, gamma);
		clock_t end = clock();
		cout << "Time lapse for computing Vesselness: ";
		cout << (float)(end - start) / CLOCKS_PER_SEC << " seconds. " << endl << endl;
		system("pause");

		// Saving VesselNess float,11
		vn_all.save( vesselness_name.str()+".vn_all", vesselness_log.str() );
		// Saving VesselNess float,5
		Data3D<Vesselness_Sig> vn_sig( vn_all );
		vn_sig.save( vesselness_name.str()+".vn_sig", vesselness_log.str() );
		// Saving VesselNess float,1
		Data3D<float> vn_float; 
		vn_sig.copyDimTo( vn_float, 0 );
		vn_float.save( vesselness_name.str()+".vn_float", vesselness_log.str() );

		// Visualize result with maximum intensity projection (MIP)
		viewer.addObject( im_short, GLViewer::Volumn::MIP );
		viewer.addObject( vn_float, GLViewer::Volumn::MIP );
		viewer.go(400, 200, 2);
		return 0;
	}
}