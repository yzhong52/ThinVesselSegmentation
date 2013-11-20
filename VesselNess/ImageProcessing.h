#pragma once

#include "stdafx.h"
#include "Image3D.h"
#include "Kernel3D.h"

class Vesselness;
class Vesselness_Sig;
class Vesselness_Nor;
class Vesselness_All;

namespace ImageProcessing
{
	///////////////////////////////////////////////////////////////////////////
	// Image Histogram
	void histogram( Data3D<short>& imageData, 
		Mat_<double>& range, Mat_<double>& hist, int number_of_bins = 512 );
	Mat histogram_with_opencv( Data3D<short>& imageData, int number_of_bins = 512 );
	void histogram_for_slice( Image3D<short>& imageData );

	///////////////////////////////////////////////////////////////////////////
	// convolution of 3d datas
	template<typename T1, typename T2, typename T3> 
	void conv3( const Data3D<T1>& src, Data3D<T2>& dst, const Kernel3D<T3>& kernel );

	///////////////////////////////////////////////////////////////////////////
	// gaussian blur 3d
	template<typename T1, typename T2> 
	bool GaussianBlur3D( const Data3D<T1>& src, Data3D<T2>& dst, int ksize, double sigma );

	///////////////////////////////////////////////////////////////////////////
	// median filter
	template<typename T1, typename T2> 
	bool medianBlur3D( const Data3D<T1>& src, Data3D<T2>& dst, int ksize);

	///////////////////////////////////////////////////////////////////////////
	// mean filter
	template<typename T1, typename T2> 
	bool meanBlur3D( const Data3D<T1>& src, Data3D<T2>& dst, int ksize);

	///////////////////////////////////////////////////////////////////////////
	// Convolution in 3 phrase
	//template<typename T1, typename T2 >
	//bool filter3D( const Data3D<T1>& src, Data3D<T2>& dst, const Kernel3D<T3>& k );
	template<typename T1, typename T2, typename T3 >
	bool filter3D_X( const Data3D<T1>& src, Data3D<T2>& dst, const Kernel3D<T3>& kx );
	template<typename T1, typename T2, typename T3 >
	bool filter3D_Y( const Data3D<T1>& src, Data3D<T2>& dst, const Kernel3D<T3>& ky );
	template<typename T1, typename T2, typename T3 >
	bool filter3D_Z( const Data3D<T1>& src, Data3D<T2>& dst, const Kernel3D<T3>& kz );

	///////////////////////////////////////////////////////////////////////////
	// show image with normalization
	void imNormShow( const string& window_name, const Mat& im );

	///////////////////////////////////////////////////////////////////////////
	// normalize the data
	template<typename T>
	void normalize( Data3D<T>& data, T norm_max = 1);

	// normalize the data
	template<typename T>
	void quad_normalize( Data3D<T>& data, T norm_max = 1);

	///////////////////////////////////////////////////////////////////////////
	// threshold the data
	template<typename T>
	void threshold( const Data3D<T>& src, Data3D<unsigned char>& dst, T thresh ){
		int x,y,z;
		dst.reset( src.get_size() );
		for(z=0;z<src.SZ();z++) for (y=0;y<src.SY();y++) for(x=0;x<src.SX();x++) {
			dst.at(x,y,z) = src.at(x,y,z) > thresh ? 255 : 0;
		}
	}
	template<typename T>
	void threshold( const Data3D<T>& src, Data3D<T>& dst, T thresh ){
		int x,y,z;
		dst.reset( src.get_size() );
		for(z=0;z<src.SZ();z++) for (y=0;y<src.SY();y++) for(x=0;x<src.SX();x++) {
			dst.at(x,y,z) = src.at(x,y,z) > thresh ? src.at(x,y,z) : 0;
		}
	}
	template<typename T>
	void threshold( Data3D<T>& src, T thresh ){
		int x,y,z;
		for(z=0;z<src.SZ();z++) for (y=0;y<src.SY();y++) for(x=0;x<src.SX();x++) {
			if( src.at(x,y,z) < thresh ) 
				memset( &src.at(x,y,z), 0, sizeof(T) );
		}
	}

	// Non-maximum suppression
	//void non_max_suppress( const Data3D<Vesselness_Sig>& src, Data3D<Vesselness_Sig>& dst );
	//void non_max_suppress( const Data3D<Vesselness_Nor>& src, Data3D<Vesselness_Sig>& dst );
	void non_max_suppress( const Data3D<Vesselness_All>& src, Data3D<Vesselness_Sig>& dst );
	void edge_tracing( Data3D<Vesselness_Sig>& src, Data3D<Vesselness_Sig>& dst, const float& thres1 = 0.85f, const float& thres2 = 0.10f );
	void dir_tracing( Data3D<Vesselness_All>& src, Data3D<Vesselness_Sig>& res_dt );
	void edge_tracing_mst( Data3D<Vesselness_All>& src, Data3D<Vesselness_Sig>& dst, const float& thres1 = 0.85f, const float& thres2 = 0.10f );
	
	// Morphological Operations
	// dilation, erosion, closing
	void dilation(Data3D<unsigned char>& src, const int& k); 
	void erosion( Data3D<unsigned char>& src, const int& k); 
	void closing( Data3D<unsigned char>& src, const int& k); 
}

namespace IP = ImageProcessing;

template<typename T1, typename T2, typename T3> 
void ImageProcessing::conv3( const Data3D<T1>& src, Data3D<T2>& dst, const Kernel3D<T3>& kernel )
{
	dst.reset( src.get_size() );
	static int x, y, z;
	for( z = 0; z < dst.get_size_z(); z++ ){
		for( y = 0; y < dst.get_size_y(); y++ ) {
			for( x = 0; x < dst.get_size_x(); x++ ){
				static int kn_x, kn_y, kn_z;
				static int im_x, im_y, im_z;
				for( kn_z = kernel.min_pos(2); kn_z < kernel.max_pos(2); kn_z++ ) {
					im_z = z + kn_z;
					if( im_z<0 || im_z>=dst.get_size_z() ) continue;
					for( kn_y = kernel.min_pos(1); kn_y < kernel.max_pos(1); kn_y++) {
						im_y = y + kn_y;
						if( im_y<0 || im_y>=dst.get_size_y() ) continue;
						for( kn_x = kernel.min_pos(0); kn_x < kernel.max_pos(0); kn_x++) {
							im_x = x + kn_x;
							if( im_x<0 || im_x>=dst.get_size_x() ) continue;
							// compute the value
							dst.at(x, y, z) += T2( src.at(im_x, im_y, im_z)*kernel.offset_at( kn_x, kn_y, kn_z ) );
						}
					}
				}		
			}
		}
	}
}


template<typename T1, typename T2> 
bool ImageProcessing::GaussianBlur3D( const Data3D<T1>& src, Data3D<T2>& dst, int ksize, double sigma ){
	smart_return_value( ksize%2!=0, "kernel size should be odd number", false );

	//////////////////////////////////////////////////////////////////////////////////////
	// Relationship between Sigma and Kernal Size (ksize)
	/////////////////////////////
	// Option I - Based on OpenCV
	//   Calculate sigma from size: 
	//         sigma = 0.3 * ( (ksize-1)/2 - 1 ) + 0.8 = 0.15*ksize + 0.35
	//   Calculate size from sigma: 
	//         ksize = ( sigma - 0.35 ) / 0.15 = 6.67 * sigma - 2.33
	//   Reference: OpenCv Documentation 2.6.4.0 getGaussianKernel
	//     http://docs.opencv.org/modules/imgproc/doc/filtering.html#creategaussianfilter
	// Option II - Based on the traditional 99.7%
	//   Calculate size from sigma: 
	//         ksize = 6 * sigma + 1
	//   Calculate sigma from ksize: 
	//         sigma = (size - 1)/6 = 0.17 * size - 0.17
	
	Mat gk = cv::getGaussianKernel( ksize, sigma, CV_64F );
	
	int hsize = ksize/2;
	smart_assert( 2*hsize+1==ksize, "Alert: Bug!" );

	Vec3i spos, epos;
	spos = Vec3i(0, 0, 0);
	epos = src.get_size();
	
	int x, y, z;
	Data3D<T2> tmp1( src.get_size() );
	Data3D<T2> tmp2( src.get_size() );
	

	// gaussian on x-direction
	for( z=spos[2]; z<epos[2]; z++ ) for( y=spos[1]; y<epos[1]; y++ ) for( x=spos[0]; x<epos[0]; x++ ){
		float sum = 0.0;
		for( int i=0; i<ksize; i++) {
			int x2 = x+i-hsize;
			if( x2<0 || x2>=epos[0] ) continue;
			tmp1.at(x, y, z) += int( gk.at<double>(i) * src.at(x2, y, z) );
			sum += (float) gk.at<double>(i);
		}
		tmp1.at(x, y, z) /= sum;
	}
	// gaussian on y-direction
	for( z=spos[2]; z<epos[2]; z++ ) for( y=spos[1]; y<epos[1]; y++ ) for( x=spos[0]; x<epos[0]; x++ ){
		float sum = 0.0f;
		for( int i=0; i<ksize; i++) {
			int y2 = y+i-hsize;
			if( y2<0 || y2>=epos[1] ) continue;
			tmp2.at(x, y, z) += int( gk.at<double>(i) * tmp1.at(x, y2, z) );
			sum += (float) gk.at<double>(i);
		}
		tmp1.at(x, y, z) /= sum;
	}
	// gaussian on z-direction
	dst.reset( src.get_size() );
	for( z=spos[2]; z<epos[2]; z++ ) for( y=spos[1]; y<epos[1]; y++ ) for( x=spos[0]; x<epos[0]; x++ ){
		float sum = 0.0f;
		for( int i=0; i<ksize; i++) {
			int z2 = z+i-hsize;
			if( z2<0 || z2>=epos[2] ) continue;
			dst.at(x, y, z) += int( gk.at<double>(i) * tmp2.at(x, y, z2) );
			sum += (float) gk.at<double>(i);
		}
		tmp1.at(x, y, z) /= sum;
	}
	return true;
}


// median filter
template<typename T1, typename T2> 
bool ImageProcessing::medianBlur3D( const Data3D<T1>& src, Data3D<T2>& dst, int ksize) {
	smart_assert(0, "Not Implemented!" );
}

// mean filter
template<typename T1, typename T2> 
bool ImageProcessing::meanBlur3D( const Data3D<T1>& src, Data3D<T2>& dst, int ksize)
{
	cout << "Blurring Image with Mean Filter..." << endl;

	Kernel3D<float> kx( Vec3i(ksize, 1, 1) );
	Kernel3D<float> ky( Vec3i(1, ksize, 1) );
	Kernel3D<float> kz( Vec3i(1, 1, ksize) );

	float kvalue = 1.0f / ksize;
	for( int i=0; i<ksize; i++ ){
		kx.at(i,0,0) = kvalue;
		ky.at(0,i,0) = kvalue;
		kz.at(0,0,i) = kvalue;
	}

	bool flag;
	Data3D<T2> tmp;
	flag = filter3D_X( src, dst, kx ); 
	if(!flag){
		cout << "Failed." << endl << endl;
		return false; 
	}

	flag = filter3D_Y( dst, tmp, ky );
	if(!flag){
		cout << "Failed." << endl << endl;
		return false; 
	}
	
	flag = filter3D_Z( tmp, dst, kz );
	if(!flag){
		cout << "Failed." << endl << endl;
		return false; 
	}

	cout << "done." << endl << endl;
	return true;
}

template<typename T1, typename T2, typename T3 >
bool ImageProcessing::filter3D_X( const Data3D<T1>& src, Data3D<T2>& dst, const Kernel3D<T3>& kx ){
	smart_return_false( (&src)!=(&dst), "src and dst should be different." );
	smart_return_false( kx.get_size_x()>0 && kx.get_size_y()==1 && kx.get_size_z()==1, 
		"kernel size should be (X,1,1) where X > 0" );
	
	dst.reset( src.get_size() );

	int x, y, z;
	const Vec3i& size = src.get_size();
	int ksize = kx.get_size_x();
	int hsize = kx.get_size_x()/2;
	for( z=0; z<size[2]; z++ ) for( y=0; y<size[1]; y++ ) for( x=0; x<size[0]; x++ ){
		for( int i=0; i<ksize; i++) {
			int x2 = x+i-hsize;
			if( x2<0 || x2>=size[0] ) continue;
			dst.at(x, y, z) += int( kx.at(i,0,0) * src.at(x2, y, z) );
		}
	}
	return true;
}

template<typename T1, typename T2, typename T3 >
bool ImageProcessing::filter3D_Y( const Data3D<T1>& src, Data3D<T2>& dst, const Kernel3D<T3>& ky ){
	smart_return_false( (&src)!=(&dst), "src and dst should be different." );
	smart_return_false( ky.get_size_x()==1 && ky.get_size_y()>0 && ky.get_size_z()==1, 
		"kernel size should be (1,Y,1) where Y > 0" );
	
	dst.reset( src.get_size() );

	int x, y, z;
	const Vec3i& size = src.get_size();
	int ksize = ky.get_size_y();
	int hsize = ky.get_size_y()/2;
	for( z=0; z<size[2]; z++ ) for( y=0; y<size[1]; y++ ) for( x=0; x<size[0]; x++ ){
		for( int i=0; i<ksize; i++) {
			int y2 = y+i-hsize;
			if( y2<0 || y2>=size[1] ) continue;
			dst.at(x, y, z) += int( ky.at(0,i,0) * src.at(x, y2, z) );
		}
	}
	return true;
}


template<typename T1, typename T2, typename T3 >
bool ImageProcessing::filter3D_Z( const Data3D<T1>& src, Data3D<T2>& dst, const Kernel3D<T3>& kz ){
	smart_return_false( (&src)!=(&dst), "src and dst should be different." );
	smart_return_false( kz.get_size_x()==1 && kz.get_size_y()==1 && kz.get_size_z()>0, 
		"kernel size should be (1,1,Z) where Z > 0 " );
	
	dst.reset( src.get_size() );

	int x, y, z;
	const Vec3i& size = src.get_size();
	int ksize = kz.get_size_z();
	int hsize = kz.get_size_z()/2;
	for( z=0; z<size[2]; z++ ) for( y=0; y<size[1]; y++ ) for( x=0; x<size[0]; x++ ){
		for( int i=0; i<ksize; i++) {
			int z2 = z+i-hsize;
			if( z2<0 || z2>=size[2] ) continue;
			dst.at(x, y, z) += int( kz.at(0,0,i) * src.at(x, y, z2) );
		}
	}
	return true;
}


// normalize the data
template<typename T>
void ImageProcessing::normalize( Data3D<T>& data, T norm_max ){
	Vec<T, 2> min_max = data.get_min_max_value();
	data.getMat() = norm_max * (data.getMat() - min_max[0]) / (min_max[1]-min_max[0]);
}

// normalize the data
template<typename T>
void ImageProcessing::quad_normalize( Data3D<T>& data, T norm_max ){
	Vec<T, 2> min_max = data.get_min_max_value();
	data.getMat() = (data.getMat() - min_max[0]) / (min_max[1]-min_max[0]);
	
	for( int z=0; z<data.get_size_z(); z++ ) {
		for( int y=0; y<data.get_size_y(); y++ ) {
			for( int x=0; x<data.get_size_x(); x++ ) {
				data.at(x, y, z) = sqrt( data.at(x, y, z) );
			}
		}
	}

	data.getMat() *= norm_max; 
}

