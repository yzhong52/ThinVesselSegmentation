#pragma once

template<typename T> class Data3D;
template<typename T> class Image3D;

namespace Validation
{
	// explore the difference between 
	// 1) Harris Detector
	// 2) Hessian Matrix
	// 3) Optimal Oriented Flux
	bool Harris_Hessian_OOP(void);

	bool Hessian_3D(void);
	bool Hessian_2D(void);
	
	bool Rings_Reduction_Polar_Coordinates( const Mat& im, Mat& dst, int wsize );
	bool Rings_Reduction_Cartecian_Coordinates( const Mat& src, Mat& dst );

	void construct_tube(  Data3D<short>& image );
	void construct_tube2( Data3D<short>& image );
}

