#pragma once

#include "stdafx.h"

template<typename T> class Data3D;
template<typename T> class Image3D;
template<typename T> class Kernel3D;

class Vesselness; 
class Vesselness_Sig; 
class Vesselness_Nor; 
class Vesselness_All; 

namespace VesselDetector
{	
	bool hessien( 
		const Data3D<short>& src, Data3D<Vesselness_Nor>& dst, 
		int ksize, float sigma, 
		float alpha, float beta, float gamma ); 
	bool hessien2( 
		const Data3D<short>& src, Data3D<Vesselness_Nor>& dst, 
		int ksize, float sigma, 
		float alpha, float beta, float gamma ); 

	// Get the eigen values and eigen vector for Hessian
	// dst[0-2]: eigenvalues - sorted as |dst[0]| < |dst[1]| < |dst[2]|
	// dst[3-5]: eigenvector 1
	// dst[6-8]: eigenvector 2
	// dst[9-11]: eigenvector 3
	bool hessien( const Data3D<short>& src, Data3D<Vec<float, 12>>& dst, 
		int ksize, float sigma, 
		float alpha, float beta, float gamma ); 
	


	int compute_vesselness( 
		const Data3D<short>& src, // INPUT
		Data3D<Vesselness_All>& dst,  // OUTPUT
		float sigma_from, float sigma_to, float sigma_step, // INPUT 
		float alpha = 1.0e-1f,	// INPUT 
		float beta  = 5.0e0f,	// INPUT 
		float gamma = 3.5e5f ); // INPUT 
	int compute_vesselness2( 
		const Data3D<short>& src, // INPUT
		Data3D<Vesselness_All>& dst,  // OUTPUT
		float sigma_from, float sigma_to, float sigma_step, // INPUT 
		float alpha = 1.0e-1f,	// INPUT 
		float beta  = 5.0e0f,	// INPUT 
		float gamma = 3.5e5f ); // INPUT 
};

namespace VD = VesselDetector;
