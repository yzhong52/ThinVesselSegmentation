#include "VesselDetector.h"
#include "Image3D.h"
#include "Kernel3D.h"
#include "ImageProcessing.h"
#include "VesselNessTypes.h"
#include "../EigenDecomp/eigen_decomp.h"

#define _CRT_SECURE_NO_DEPRECATE
#include <opencv2\core\core.hpp>



bool VesselDetector::hessien2( const Data3D<short>& src, Data3D<Vesselness_Nor>& dst, 
	int ksize, float sigma, 
	float alpha, float beta, float gamma )
{
	if( ksize!=0 && sigma<1e-3 ) {
		// sigma is note set
		sigma = 0.15f * ksize + 0.35f;
	} else if ( ksize==0 && sigma>1e-3 ) {
		// size is unset
		ksize = int( 6*sigma+1 );
		// make sure ksize is an odd number
		if ( ksize%2==0 ) ksize++;
	} else {
		cerr << "At lease ksize or sigma has to be set." << endl;
		return false;
	}

	Image3D<float> im_blur;
	bool flag = ImageProcessing::GaussianBlur3D( src, im_blur, ksize, sigma );
	smart_return_value( flag, "Gaussian Blur Failed.", false );
	
	//Normalizing for different scale
	im_blur *= sigma;

	Vesselness_Nor temp; 
	temp.rsp = 0.0f; 
	dst.reset( im_blur.get_size(), temp ); 

	int x, y, z; 
	for( z = 2; z < src.get_size_z()-2; z++ ) {
		for( y = 2; y < src.get_size_y()-2; y++ ) {
			for( x = 2; x < src.get_size_x()-2; x++ ) {
				Data3D<float>& src = im_blur;

				////////////////////////////////////////////////////////////////////
				// The following are being computed in this function
				// 1) derivative of images; 
				// 2) Hessian matrix; 
				// 3) eigenvalue decomposition; 
				// 4) vesselness measure. 
				
				// 1) derivative of the image		
				float im_dx2 = -2.0f * src.at(x,y,z) + src.at(x-1,y,z) + src.at(x+1,y,z);
				float im_dy2 = -2.0f * src.at(x,y,z) + src.at(x,y-1,z) + src.at(x,y+1,z);
				float im_dz2 = -2.0f * src.at(x,y,z) + src.at(x,y,z-1) + src.at(x,y,z+1);
				// 1) derivative of the image (alternative) 		
				//float im_dx2 = -0.5f * src.at(x,y,z) + 0.25f * src.at(x-2,y,z) + 0.25f * src.at(x+2,y,z);
				//float im_dy2 = -0.5f * src.at(x,y,z) + 0.25f * src.at(x,y-2,z) + 0.25f * src.at(x,y+2,z);
				//float im_dz2 = -0.5f * src.at(x,y,z) + 0.25f * src.at(x,y,z-2) + 0.25f * src.at(x,y,z+2);

				float im_dxdy = ( 
					+ src.at(x-1, y-1, z) 
					+ src.at(x+1, y+1, z) 
					- src.at(x-1, y+1, z) 
					- src.at(x+1, y-1, z) ) * 0.25f; 
				float im_dxdz = ( 
					+ src.at(x-1, y, z-1) 
					+ src.at(x+1, y, z+1)
					- src.at(x+1, y, z-1) 
					- src.at(x-1, y, z+1) ) * 0.25f; 
				float im_dydz = ( 
					+ src.at(x, y-1, z-1) 
					+ src.at(x, y+1, z+1)
					- src.at(x, y+1, z-1) 
					- src.at(x, y-1, z+1) ) * 0.25f; 

				// 3) eigenvalue decomposition
				// Reference: http://en.wikipedia.org/wiki/Eigenvalue_algorithm#3.C3.973_matrices
				// Given a real symmetric 3x3 matrix A, compute the eigenvalues
				const float A[6] = {im_dx2, im_dxdy, im_dxdz, im_dy2, im_dydz, im_dz2}; 

				float eigenvalues[3]; 
				float eigenvectors[3][3]; 
				eigen_decomp( A, eigenvalues, eigenvectors );

				// order eigenvalues so that |lambda1| < |lambda2| < |lambda3| 
				int i=0, j=1, k=2;
				if( abs(eigenvalues[i]) > abs(eigenvalues[j]) ) std::swap( i, j );
				if( abs(eigenvalues[i]) > abs(eigenvalues[k]) ) std::swap( i, k );
				if( abs(eigenvalues[j]) > abs(eigenvalues[k]) ) std::swap( j, k );
				
				Vesselness_Nor vn_Nor; 
				// vesselness value
				if( eigenvalues[j] > 0 || eigenvalues[k] > 0 ) {
					vn_Nor.rsp = 0.0f;
				} else {
					float lmd1 = abs( eigenvalues[i] );
					float lmd2 = abs( eigenvalues[j] );
					float lmd3 = abs( eigenvalues[k] );
					
					float A = (lmd3>1e-5) ? lmd2/lmd3 : 0;
					float B = (lmd2*lmd3>1e-5) ? lmd1 / sqrt( lmd2*lmd3 ) : 0;
					float S = sqrt( lmd1*lmd1 + lmd2*lmd2 + lmd3*lmd3 );
					vn_Nor.rsp = ( 1.0f-exp(-A*A/alpha) )* exp( B*B/beta ) * ( 1-exp(-S*S/gamma) );
				}
				
				// orientation of vesselness
				for( int d=0; d<3; d++ ) {
					vn_Nor.dir[d]        = eigenvectors[i][d];
					vn_Nor.normals[0][d] = eigenvectors[j][d];
					vn_Nor.normals[1][d] = eigenvectors[k][d];
				}
				
				// copy the value to our dst matrix
				dst.at(x, y, z) = vn_Nor; 
			}
		}
	}
	return true;
}



int VesselDetector::compute_vesselness2( 
	const Data3D<short>& src,							// INPUT
	Data3D<Vesselness_All>& dst,						// OUTPUT
	float sigma_from, float sigma_to, float sigma_step, // INPUT 
	float alpha, float beta, float gamma )				// INPUT
{
	cout << "Computing Vesselness, it will take a while... " << endl;
	cout << "Vesselness will be computed from sigma = " << sigma_from << " to sigma = " << sigma_to << endl;
	
	Data3D<Vesselness_Nor> vn;
	dst.reset( src.get_size() ); // reszie data, and it will also be clear to zero

	// Error for input parameters
	smart_return_value( sigma_from < sigma_to, 
		"sigma_from should be smaller than sigma_to ", 0 );
	smart_return_value( sigma_step > 0, 
		"sigma_step should be greater than 0 ", 0 );

	int x,y,z;
	float max_sigma = sigma_from;
	float min_sigma = sigma_to;

	for( float sigma = sigma_from; sigma < sigma_to; sigma += sigma_step ){
		cout << '\r' << "Vesselness for sigma = " << sigma << "         ";
		VesselDetector::hessien2( src, vn, 0, sigma, alpha, beta, gamma );
		// compare the response, if it is greater, copy it to our dst
		int margin = 1; // int( ceil(3 * sigma) );
		//if( margin % 2 == 0 )  margin++;
		for( z=margin; z<src.get_size_z()-margin; z++ ) {
			for( y=margin; y<src.get_size_y()-margin; y++ ) {
				for( x=margin; x<src.get_size_x()-margin; x++ ) {
					if( dst.at(x, y, z).rsp < vn.at(x, y, z).rsp ) {
						dst.at(x, y, z) = Vesselness_All( vn.at(x, y, z), sigma );
						max_sigma = max( sigma, max_sigma );
						min_sigma = min( sigma, min_sigma );
					}
				}
			}
		}
	}

	cout << endl << "The minimum and maximum sigmas used for vesselness: " << min_sigma << ", " << max_sigma << endl;
	cout << "done. " << endl << endl;

	return 0;
}
