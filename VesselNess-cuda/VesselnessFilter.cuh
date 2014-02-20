#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Data3D.h"
#include "ImageProcessing.h"


namespace VesselnessFilterGPU{
	int compute_vesselness( 
		const Data3D<short>& src, // INPUT
		Data3D<float>& dst,  // OUTPUT
		float sigma_from, float sigma_to, float sigma_step, // INPUT 
		float alpha = 1.0e-1f,	// INPUT 
		float beta  = 5.0e0f,	// INPUT 
		float gamma = 3.5e5f ); // INPUT 
}

namespace VFG = VesselnessFilterGPU; 


int VesselnessFilterGPU::compute_vesselness( 
		const Data3D<short>& src, // INPUT
		Data3D<float>& dst,  // OUTPUT
		float sigma_from, float sigma_to, float sigma_step, // INPUT 
		float alpha,	// INPUT parameter 
		float beta,		// INPUT parameter
		float gamma     // INPUT parameter 
{
	
	return 0; 
}
