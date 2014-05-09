Vesselness Measure/Filter
========================

This is a cpp implementation of Vesselness Measure for 3D volume based on the following paper. 

- Frangi, Alejandro F., et al. "Multiscale vessel enhancement filtering." Medical Image Computing and Computer-Assisted Interventation—MICCAI’98. Springer Berlin Heidelberg, 1998. 130-137.

Two sample code is provided in main.cpp
- computing vesselness measure
- extracting the vessel centrelines with non-maximum suppression

Example Usage
---------
		// Sigma: Parameters for Vesselness
		// [sigma_from, sigma_to]: the potential size rang of the vessels
		// sigma_step: precision of computation
		float sigma_from = 1.0f;
		float sigma_to = 8.10f;
		float sigma_step = 0.5f;
		// Parameters for vesselness, please refer to Frangi's papaer 
		// or this [blog](http://yzhong.co/?p=351)
		float alpha = 1.0e-1f;	
		float beta  = 5.0e0f;
		float gamma = 3.5e5f;

		// laoding data
		Data3D<short> im_short;
		bool flag = im_short.load( "../data/" + dataname + ".data" );
		if(!flag) return 0;
		
		// Compute Vesselness
		Data3D<Vesselness_Sig> vn_sig; 
		VesselDetector::compute_vesselness( im_short, vn_sig, 
			sigma_from, sigma_to, sigma_step,
			alpha, beta, gamma );
			
		// Saving Data
		vn_sig.save( "../temp/" + dataname + ".vn_sig" );

		// If you want to visulize the data using Maximum-Intensity Projection
		if( isDisplay ) {
			viewer.addObject( vn_sig,  GLViewer::Volumn::MIP );
			viewer.addDiretionObject( vn_sig );
			viewer.go(600, 400, 2);
		}
