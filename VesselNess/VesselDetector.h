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

	bool hessien( 
		const Data3D<short>& src, Data3D<Vesselness>& dst, 
		int ksize, float sigma, 
		float alpha, float beta, float gamma );

	bool hessien( 
		const Data3D<short>& src, Data3D<float>& dst, 
		int ksize, float sigma, 
		float alpha, float beta, float gamma );

	int compute_vesselness( 
		const Data3D<short>& src, // INPUT
		Data3D<Vesselness_All>& dst,  // OUTPUT
		float sigma_from, float sigma_to, float sigma_step ); // INPUT

	int compute_vesselness( 
		const Data3D<short>& src, // INPUT
		Data3D<Vesselness>& dst,  // OUTPUT
		float sigma_from, float sigma_to, float sigma_step ); // INPUT

	int compute_vesselness( 
		const Data3D<short>& src, // INPUT
		Data3D<float>& dst,  // OUTPUT
		float sigma_from, float sigma_to, float sigma_step ); // INPUT
};

namespace VD = VesselDetector;
