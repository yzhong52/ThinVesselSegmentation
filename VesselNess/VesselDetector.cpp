#include "VesselDetector.h"
#include "Image3D.h"
#include "Kernel3D.h"
#include "ImageProcessing.h"
#include "Vesselness.h"

#include <opencv2\core\core.hpp>

bool VesselDetector::hessien( const Data3D<short>& src, Data3D<Vesselness_Nor>& dst, 
	int ksize, float sigma, 
	float alpha, float beta, float gamma )
{
	if( ksize!=0 && sigma<1e-3 ) {
		// sigma is unset
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
	
	/*if( sigma<0.5 ) {  im_blur *= 0.25 + sigma/2; }
	else */im_blur *= sigma;
	
	Kernel3D<float> dx = Kernel3D<float>::dx();
	Kernel3D<float> dy = Kernel3D<float>::dy();
	Kernel3D<float> dz = Kernel3D<float>::dz();

	// First Order Derivative
	Image3D<int> im_dx, im_dy, im_dz;
	ImageProcessing::conv3( im_blur, im_dx, dx );
	ImageProcessing::conv3( im_blur, im_dy, dy );
	ImageProcessing::conv3( im_blur, im_dz, dz );

	// Second Order Derivative
	Image3D<float> im_dx2, im_dy2, im_dz2;
	Image3D<float> im_dxdy, im_dxdz, im_dydz;
	ImageProcessing::conv3( im_dx, im_dx2, dx );
	ImageProcessing::conv3( im_dy, im_dy2, dy );
	ImageProcessing::conv3( im_dz, im_dz2, dz );
	ImageProcessing::conv3( im_dx, im_dxdy, dy );
	ImageProcessing::conv3( im_dx, im_dxdz, dz );
	ImageProcessing::conv3( im_dy, im_dydz, dz );

	dst.reset( src.get_size() );

	int x, y, z;
	Vesselness_Nor vn_Nor;
	for( z=0; z<src.get_size_z(); z++ ) {
		for( y=0; y<src.get_size_y(); y++ ) {
			for( x=0; x<src.get_size_x(); x++ ) {
				// construct the harris matrix
				Mat hessian( 3, 3, CV_32F );
				hessian.at<float>(0, 0) = im_dx2.at(x,y,z);
				hessian.at<float>(1, 1) = im_dy2.at(x,y,z);
				hessian.at<float>(2, 2) = im_dz2.at(x,y,z);
				hessian.at<float>(1, 0) = hessian.at<float>(0, 1) = im_dxdy.at(x,y,z);
				hessian.at<float>(2, 0) = hessian.at<float>(0, 2) = im_dxdz.at(x,y,z);
				hessian.at<float>(2, 1) = hessian.at<float>(1, 2) = im_dydz.at(x,y,z);
				
				// calculate the eigen values
				Mat eigenvalues, eigenvectors;
				eigen( hessian, eigenvalues, eigenvectors ); 
				// order eigenvalues so that |lambda1| < |lambda2| < |lambda3| 
				int i=0, j=1, k=2;
				if( abs(eigenvalues.at<float>(i)) > abs(eigenvalues.at<float>(j)) ) std::swap( i, j );
				if( abs(eigenvalues.at<float>(i)) > abs(eigenvalues.at<float>(k)) ) std::swap( i, k );
				if( abs(eigenvalues.at<float>(j)) > abs(eigenvalues.at<float>(k)) ) std::swap( j, k );
				
				// vesselness value
				if( eigenvalues.at<float>(j) > 0 || eigenvalues.at<float>(k) > 0 ) {
					vn_Nor.rsp = 0.0f;
				} else {
					float lmd1 = abs( eigenvalues.at<float>(i) );
					float lmd2 = abs( eigenvalues.at<float>(j) );
					float lmd3 = abs( eigenvalues.at<float>(k) );
					float A = lmd2 / lmd3;
					float B = lmd1 / sqrt( lmd2*lmd3 );
					float S = sqrt( lmd1*lmd1 + lmd2*lmd2 + lmd3*lmd3 );
					vn_Nor.rsp = ( 1.0f-exp(-A*A/alpha) )* exp( B*B/beta ) * ( 1-exp(-S*S/gamma) );
					// vn_Nor.rsp = ( 1.0f-exp(-A*A/alpha) );
					// vn_Nor.rsp = exp( B*B/beta );
					// vn_Nor.rsp = ( 1-exp(-S*S/gamma) );
					// float D = lmd1*lmd2*lmd3;
					// vn_Nor.rsp = ( 1-exp(-D/gamma) );
				}
				
				// orientation of vesselness
				for( int d=0; d<3; d++ ) {
					vn_Nor.dir[d] = eigenvectors.at<float>(i, d);
					vn_Nor.normals[0][d] = eigenvectors.at<float>(j, d);
					vn_Nor.normals[1][d] = eigenvectors.at<float>(k, d);
				}
				
				// copy the value to our dst matrix
				dst.at(x, y, z) = vn_Nor; 
			}
		}
	}
	return true;
}



bool VesselDetector::hessien( const Data3D<short>& src, Data3D<Vesselness>& dst, 
	int ksize, float sigma, 
	float alpha, float beta, float gamma )
{
	if( ksize!=0 && sigma<1e-3 ) {
		// sigma is unset
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
	
	/*if( sigma<0.5 ) {  im_blur *= 0.25 + sigma/2; }
	else */im_blur *= sigma;
	
	Kernel3D<float> dx = Kernel3D<float>::dx();
	Kernel3D<float> dy = Kernel3D<float>::dy();
	Kernel3D<float> dz = Kernel3D<float>::dz();

	// First Order Derivative
	Image3D<int> im_dx, im_dy, im_dz;
	ImageProcessing::conv3( im_blur, im_dx, dx );
	ImageProcessing::conv3( im_blur, im_dy, dy );
	ImageProcessing::conv3( im_blur, im_dz, dz );

	// Second Order Derivative
	Image3D<float> im_dx2, im_dy2, im_dz2;
	Image3D<float> im_dxdy, im_dxdz, im_dydz;
	ImageProcessing::conv3( im_dx, im_dx2, dx );
	ImageProcessing::conv3( im_dy, im_dy2, dy );
	ImageProcessing::conv3( im_dz, im_dz2, dz );
	ImageProcessing::conv3( im_dx, im_dxdy, dy );
	ImageProcessing::conv3( im_dx, im_dxdz, dz );
	ImageProcessing::conv3( im_dy, im_dydz, dz );

	dst.reset( src.get_size() );

	int x, y, z;
	for( z=0; z<src.get_size_z(); z++ ) {
		for( y=0; y<src.get_size_y(); y++ ) {
			for( x=0; x<src.get_size_x(); x++ ) {
				// construct the harris matrix
				Mat hessian( 3, 3, CV_32F );
				hessian.at<float>(0, 0) = im_dx2.at(x,y,z);
				hessian.at<float>(1, 1) = im_dy2.at(x,y,z);
				hessian.at<float>(2, 2) = im_dz2.at(x,y,z);
				hessian.at<float>(1, 0) = hessian.at<float>(0, 1) = im_dxdy.at(x,y,z);
				hessian.at<float>(2, 0) = hessian.at<float>(0, 2) = im_dxdz.at(x,y,z);
				hessian.at<float>(2, 1) = hessian.at<float>(1, 2) = im_dydz.at(x,y,z);
				
				// calculate the eigen values
				Mat eigenvalues, eigenvectors;
				eigen( hessian, eigenvalues, eigenvectors ); 
				// order eigenvalues so that |lambda1| < |lambda2| < |lambda3| 
				int i=0, j=1, k=2;
				if( abs(eigenvalues.at<float>(i)) > abs(eigenvalues.at<float>(j)) ) std::swap( i, j );
				if( abs(eigenvalues.at<float>(i)) > abs(eigenvalues.at<float>(k)) ) std::swap( i, k );
				if( abs(eigenvalues.at<float>(j)) > abs(eigenvalues.at<float>(k)) ) std::swap( j, k );
				
				// vesselness value
				if( eigenvalues.at<float>(j) > 0 || eigenvalues.at<float>(k) > 0 ) {
					dst.at(x, y, z).rsp = 0.0f;
				} else {
					float lmd1 = abs( eigenvalues.at<float>(i) );
					float lmd2 = abs( eigenvalues.at<float>(j) );
					float lmd3 = abs( eigenvalues.at<float>(k) );
					float A = lmd2 / lmd3;
					float B = lmd1 / sqrt( lmd2*lmd3 );
					float S = sqrt( lmd1*lmd1 + lmd2*lmd2 + lmd3*lmd3 );
					dst.at(x, y, z).rsp = ( 1.0f-exp(-A*A/alpha) )* exp( B*B/beta ) * ( 1-exp(-S*S/gamma) );
				}
				
				// orientation of vesselness
				for( int d=0; d<3; d++ ) {
					dst.at(x, y, z).dir[d] = eigenvectors.at<float>(i, d);
				}
			}
		}
	}
	return true;
}



bool VesselDetector::hessien( const Data3D<short>& src, Data3D<float>& dst, 
	int ksize, float sigma, 
	float alpha, float beta, float gamma )
{
	if( ksize!=0 && sigma<1e-3 ) {
		// sigma is unset
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
	
	/*if( sigma<0.5 ) {  im_blur *= 0.25 + sigma/2; }
	else */im_blur *= sigma;
	
	Kernel3D<float> dx = Kernel3D<float>::dx();
	Kernel3D<float> dy = Kernel3D<float>::dy();
	Kernel3D<float> dz = Kernel3D<float>::dz();

	// First Order Derivative
	Image3D<int> im_dx, im_dy, im_dz;
	ImageProcessing::conv3( im_blur, im_dx, dx );
	ImageProcessing::conv3( im_blur, im_dy, dy );
	ImageProcessing::conv3( im_blur, im_dz, dz );

	// Second Order Derivative
	Image3D<float> im_dx2, im_dy2, im_dz2;
	Image3D<float> im_dxdy, im_dxdz, im_dydz;
	ImageProcessing::conv3( im_dx, im_dx2, dx );
	ImageProcessing::conv3( im_dy, im_dy2, dy );
	ImageProcessing::conv3( im_dz, im_dz2, dz );
	ImageProcessing::conv3( im_dx, im_dxdy, dy );
	ImageProcessing::conv3( im_dx, im_dxdz, dz );
	ImageProcessing::conv3( im_dy, im_dydz, dz );

	dst.reset( src.get_size() );

	int x, y, z;
	for( z=0; z<src.get_size_z(); z++ ) {
		for( y=0; y<src.get_size_y(); y++ ) {
			for( x=0; x<src.get_size_x(); x++ ) {
				// construct the harris matrix
				Mat hessian( 3, 3, CV_32F );
				hessian.at<float>(0, 0) = im_dx2.at(x,y,z);
				hessian.at<float>(1, 1) = im_dy2.at(x,y,z);
				hessian.at<float>(2, 2) = im_dz2.at(x,y,z);
				hessian.at<float>(1, 0) = hessian.at<float>(0, 1) = im_dxdy.at(x,y,z);
				hessian.at<float>(2, 0) = hessian.at<float>(0, 2) = im_dxdz.at(x,y,z);
				hessian.at<float>(2, 1) = hessian.at<float>(1, 2) = im_dydz.at(x,y,z);
				
				// calculate the eigen values
				Mat eigenvalues, eigenvectors;
				eigen( hessian, eigenvalues, eigenvectors ); 
				// order eigenvalues so that |lambda1| < |lambda2| < |lambda3| 
				int i=0, j=1, k=2;
				if( abs(eigenvalues.at<float>(i)) > abs(eigenvalues.at<float>(j)) ) std::swap( i, j );
				if( abs(eigenvalues.at<float>(i)) > abs(eigenvalues.at<float>(k)) ) std::swap( i, k );
				if( abs(eigenvalues.at<float>(j)) > abs(eigenvalues.at<float>(k)) ) std::swap( j, k );
				
				// vesselness value
				if( eigenvalues.at<float>(j) > 0 || eigenvalues.at<float>(k) > 0 ) {
					dst.at(x, y, z) = 0.0f;
				} else {
					float lmd1 = abs( eigenvalues.at<float>(i) );
					float lmd2 = abs( eigenvalues.at<float>(j) );
					float lmd3 = abs( eigenvalues.at<float>(k) );
					float A = lmd2 / lmd3;
					float B = lmd1 / sqrt( lmd2*lmd3 );
					float S = sqrt( lmd1*lmd1 + lmd2*lmd2 + lmd3*lmd3 );
					dst.at(x, y, z) = ( 1.0f-exp(-A*A/alpha) )* exp( B*B/beta ) * ( 1-exp(-S*S/gamma) );
				}
			}
		}
	}
	return true;
}


int VesselDetector::compute_vesselness( 
	const Data3D<short>& src, // INPUT
	Data3D<Vesselness_All>& dst,  // OUTPUT
	float sigma_from, float sigma_to, float sigma_step ) // INPUT
{
	cout << "Computing Vesselness, it will take a while... " << endl;
	cout << "Vesselness will be computed from sigma = " << sigma_from << " to sigma = " << sigma_to << endl;
	
	Data3D<Vesselness_Nor> vn;
	dst.reset( src.get_size() ); // data will be clear to zero


	smart_return_value( sigma_from < sigma_to, "sigma_from should be smaller than sigma_to ", 0 );
	smart_return_value( sigma_step > 0, "sigma_step should be greater than 0 ", 0 );

	
	// The BIGGER the Value the less sensitive the term is
	const float alpha = 1.0e-1f;     // A = lmd2 / lmd3;
	// The SMALLER the Value the less sensitive the term is
	const float beta  = 5.0e0f;     // B = lmb1 / sqrt( lmb2 * lmb3 );
	// The BIGGER the Value the less sensitive the term is
	const float gamma = 3.5e5;      // S = sqrt( lmd1*lmd1 + lmd2*lmd2 + lmd3*lmd3 );
	// vn_Nor.rsp = ( 1.0f-exp(-A*A/alpha) )* exp( B*B/beta ) * ( 1-exp(-S*S/gamma) );
	
	int x,y,z;
	float max_sigma = sigma_from;
	float min_sigma = sigma_to;
	for( float sigma = sigma_from; sigma < sigma_to; sigma += sigma_step ){
		cout << '\r' << "Vesselness for sigma = " << sigma;
		VesselDetector::hessien( src, vn, 0, sigma, alpha, beta, gamma );
		// compare the response, if it is greater, copy it to our dst
		int margin = int( ceil(6 * sigma_to + 2) );
		if( margin % 2 == 0 )  margin++;
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



int VesselDetector::compute_vesselness( 
	const Data3D<short>& src, // INPUT
	Data3D<Vesselness>& dst,  // OUTPUT
	float sigma_from, float sigma_to, float sigma_step ) // INPUT
{
	cout << "Computing Vesselness, it will take a while... " << endl;
	cout << "Vesselness will be computed from sigma = " << sigma_from << " to sigma = " << sigma_to << endl;
	
	Data3D<Vesselness> vn;
	dst.reset( src.get_size() ); // data will be clear to zero


	smart_return_value( sigma_from < sigma_to, "sigma_from should be smaller than sigma_to ", 0 );
	smart_return_value( sigma_step > 0, "sigma_step should be greater than 0 ", 0 );

	
	// The BIGGER the Value the less sensitive the term is
	const float alpha = 1.0e-1f;     // A = lmd2 / lmd3;
	// The SMALLER the Value the less sensitive the term is
	const float beta  = 5.0e0f;     // B = lmb1 / sqrt( lmb2 * lmb3 );
	// The BIGGER the Value the less sensitive the term is
	const float gamma = 3.5e5;      // S = sqrt( lmd1*lmd1 + lmd2*lmd2 + lmd3*lmd3 );
	// vn_Nor.rsp = ( 1.0f-exp(-A*A/alpha) )* exp( B*B/beta ) * ( 1-exp(-S*S/gamma) );
	
	int x,y,z;
	float max_sigma = sigma_from;
	float min_sigma = sigma_to;
	for( float sigma = sigma_from; sigma < sigma_to; sigma += sigma_step ){
		cout << '\r' << "Vesselness for sigma = " << sigma;
		VesselDetector::hessien( src, vn, 0, sigma, alpha, beta, gamma );
		// compare the response, if it is greater, copy it to our dst
		int margin = int( ceil(6 * sigma_to + 2) );
		if( margin % 2 == 0 )  margin++;
		for( z=margin; z<src.get_size_z()-margin; z++ ) {
			for( y=margin; y<src.get_size_y()-margin; y++ ) {
				for( x=margin; x<src.get_size_x()-margin; x++ ) {
					if( dst.at(x, y, z).rsp < vn.at(x, y, z).rsp ) {
						dst.at(x, y, z) = vn.at(x, y, z);
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



int VesselDetector::compute_vesselness( 
	const Data3D<short>& src, // INPUT
	Data3D<float>& dst,  // OUTPUT
	float sigma_from, float sigma_to, float sigma_step ) // INPUT
{
	cout << "Computing Vesselness, it will take a while... " << endl;
	cout << "Vesselness will be computed from sigma = " << sigma_from << " to sigma = " << sigma_to << endl;
	
	Data3D<float> temp_rep;
	dst.reset( src.get_size() ); // data will be clear to zero


	smart_return_value( sigma_from < sigma_to, "sigma_from should be smaller than sigma_to ", 0 );
	smart_return_value( sigma_step > 0, "sigma_step should be greater than 0 ", 0 );

	
	// The BIGGER the Value the less sensitive the term is
	const float alpha = 1.0e-1f;     // A = lmd2 / lmd3;
	// The SMALLER the Value the less sensitive the term is
	const float beta  = 5.0e0f;     // B = lmb1 / sqrt( lmb2 * lmb3 );
	// The BIGGER the Value the less sensitive the term is
	const float gamma = 3.5e5;      // S = sqrt( lmd1*lmd1 + lmd2*lmd2 + lmd3*lmd3 );
	// vn_Nor.rsp = ( 1.0f-exp(-A*A/alpha) )* exp( B*B/beta ) * ( 1-exp(-S*S/gamma) );
	
	int x,y,z;
	float max_sigma = sigma_from;
	float min_sigma = sigma_to;
	for( float sigma = sigma_from; sigma < sigma_to; sigma += sigma_step ){
		cout << '\r' << "Vesselness for sigma = " << sigma;
		VesselDetector::hessien( src, temp_rep, 0, sigma, alpha, beta, gamma );
		// compare the response, if it is greater, copy it to our dst
		int margin = int( ceil(6 * sigma_to + 2) );
		if( margin % 2 == 0 )  margin++;
		for( z=margin; z<src.get_size_z()-margin; z++ ) {
			for( y=margin; y<src.get_size_y()-margin; y++ ) {
				for( x=margin; x<src.get_size_x()-margin; x++ ) {
					if( dst.at(x, y, z) < temp_rep.at(x, y, z) ) {
						dst.at(x, y, z) = temp_rep.at(x, y, z);
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
