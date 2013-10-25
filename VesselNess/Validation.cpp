#include "stdafx.h"
#include "Validation.h"

#define _USE_MATH_DEFINES
#include <math.h>

// Visualize vesselness by plot
// each row the vesselness represent one specific setting when generating the vesselness
// value. For example, different kernal size. 
void visualize_eigens(const string& name, const Mat& vesselbackground, vector<Mat>& vesselness_group, int im_height = 400, float scale = 2.0f)
{
	// Maximum number of groups supported by the function
	static const int MAX_GROUP = 6;

	// magin of the visualization
	static const float margin = 0.05f;

	int num_group = (int) vesselness_group.size();
	smart_return( num_group>=0 && num_group<=MAX_GROUP, "Cannot handle that many number of groups" );

	int im_width = vesselness_group[0].cols;
	// Make sure every group has the same width
	for( int i=1; i<num_group; i++ ){
		if( vesselness_group[i].cols!=im_width ){
			cout << "vesselness cols do not mathc match" << endl;
			return;
		}
	}

	// Set the color the lines for each group
	// Maximum number of groups is 6. 
	vector<vector<Scalar>> colors_group;
	static const bool bits[6][3] = {
		{1,0,0}, {0,0,1}, {0,1,0}, 
		{1,1,0}, {0,1,1}, {1,0,1}  
	};
	for( int i=0; i<num_group; i++ ) {
		vector<Scalar> colors;
		Mat& vesselness = vesselness_group[i];
		for( int j=0; j<vesselness.rows; j++ ) {
			int brightness = 200*j/vesselness.rows + 55;
			Scalar c;
			c.val[0] = bits[i][0] ? 255 : brightness;
			c.val[1] = bits[i][1] ? 255 : brightness;
			c.val[2] = bits[i][2] ? 255 : brightness;
			colors.push_back( c );
		}
		colors_group.push_back( colors );
	}


	// find the maximum and minimum vesselness
	Point minLoc, maxLoc;
	// for group 0
	minMaxLoc( vesselness_group[0], NULL, NULL, &minLoc, &maxLoc);
	float max_value = vesselness_group[0].at<float>( maxLoc );
	float min_value = vesselness_group[0].at<float>( minLoc );
	// for other groups
	for( int i=1; i<num_group; i++ ){
		minMaxLoc( vesselness_group[i], NULL, NULL, &minLoc, &maxLoc);
		max_value = std::max( vesselness_group[i].at<float>( maxLoc ), max_value );
		min_value = std::min( vesselness_group[i].at<float>( minLoc ), min_value );
	}
	float max_min_gap = max_value - min_value;

	// vissualize the vesselness
	Mat im_vesselness( int( im_height*scale ), int( im_width*scale), 
		CV_8UC3, Scalar(0, 0, 0) );

	// draw the background
	for( int i=0; i<vesselbackground.cols; i++ ){
		unsigned char c = vesselbackground.at<unsigned char>( i )/5;
		Scalar color( c, c, c );
		line( im_vesselness, 
			Point(i, 0)*scale, 
			Point(i, im_height-1)*scale, 
			color, 1, CV_AA, 0 );
	}

	for( int g=0; g<num_group; g++ ) {
		Mat& vesselness = vesselness_group[g];
		vector<Scalar>& colors = colors_group[g];

		for( int i=0; i<vesselness.rows; i++ ) for( int j=1; j < vesselness.cols; j++ )
		{
			float v1 = vesselness.at<float>(i, j-1);
			float v2 = vesselness.at<float>(i, j);
			// Vesselness from Harris Detector
			Point p1, p2;
			p1.x = int( (j-1) * scale );
			p1.y = int( im_height*scale * ( margin + (1-2*margin)*(1.0 - (v1-min_value)/max_min_gap ) ) );
			p2.x = int( j * scale );
			p2.y = int( im_height*scale * ( margin + (1-2*margin)*(1.0 - (v2-min_value)/max_min_gap ) ) );

			line( im_vesselness, p1, p2, colors[i], 1, CV_AA );
		}
	}

	// show result in window and save to file
	imshow( name.c_str(), im_vesselness );
	imwrite( (name+".png").c_str(), im_vesselness );
}

void visualize_eigens(const string& name, const Mat& vesselbackground, Mat vesselness, int im_height = 600){
	vector<Mat> vesselness_group;
	vesselness_group.push_back( vesselness );
	visualize_eigens( name, vesselbackground, vesselness_group, im_height);
}

bool Validation::Harris_Hessian_OOP(void)
{
	// loading image
	Mat src = imread( "images/vessels.bmp");
	if( !src.data ){ 
		cout << "Image not found..." << endl;
		return false; 
	}

	// convert form CV_8UC3 to CV_8U
	Mat src_gray;
	cvtColor( src, src_gray, CV_RGB2GRAY ); 
	
	// Image gradient along x, y direction
	Mat Ix, Iy;
	Sobel( src_gray, Ix, CV_32F, 1, 0, 1 );
	Sobel( src_gray, Iy, CV_32F, 0, 1, 1 );
	// Second order derivative of the image
	Mat Ixx, Ixy, Iyy;
	Sobel( Ix, Ixx, CV_32F, 1, 0, 1 );
	Sobel( Ix, Ixy, CV_32F, 0, 1, 1 );
	Sobel( Iy, Iyy, CV_32F, 0, 1, 1 );
	// Prepared value of Harris Detector [Ix2 IxIy; IyIx Iy2]
	Mat IxIx, IxIy, IyIy;
	multiply( Ix, Ix, IxIx );
	multiply( Ix, Iy, IxIy );
	multiply( Iy, Iy, IyIy );

	// derivative fileters
	Mat filter_dx = ( Mat_<float>(1,3) << -0.5, 0, 0.5 );
	Mat filter_dy = ( Mat_<float>(3,1) << -0.5, 0, 0.5 );

	// Calculate the vesselness response
	const int NUM = 1;

	// Eigenvalues result will be stored in these matrix
	Mat harris_eigenvalue1( NUM, src.cols, CV_32F );
	Mat harris_eigenvalue2( NUM, src.cols, CV_32F );
	Mat hessian_eigenvalue1( NUM, src.cols, CV_32F );
	Mat hessian_eigenvalue2( NUM, src.cols, CV_32F );
	Mat oof_eigenvalue1( NUM, src.cols, CV_32F );
	Mat oof_eigenvalue2( NUM, src.cols, CV_32F );

	// iterator for different kernel size
	for( int ks=0; ks<NUM; ks++ ) { 

		// different kernal size 
		int s = 21+10*ks; 
		Size ksize(s, s); 
		cout << "Kernel Size: " << s << endl; 
		
		///////////////////////////////////////////////////////////////////////
		// Harris Corner Detector
		///////////////////////////////////////////////////////////////////////
		Mat harris_IxIx, harris_IxIy, harris_IyIy;
		GaussianBlur( IxIx, harris_IxIx, ksize, 0, 0 );
		GaussianBlur( IxIy, harris_IxIy, ksize, 0, 0 );
		GaussianBlur( IyIy, harris_IyIy, ksize, 0, 0 );
		// compute the vesselness
		for( int i=0; i<src.cols; i++ )
		{
			int j = src.rows/2;
			// construct the harris matrix
			Mat harris( 2, 2, CV_32F );
			harris.at<float>(0, 0) = harris_IxIx.at<float>(j, i);
			harris.at<float>(1, 0) = harris_IxIy.at<float>(j, i);
			harris.at<float>(0, 1) = harris_IxIy.at<float>(j, i);
			harris.at<float>(1, 1) = harris_IyIy.at<float>(j, i);
			// calculate the eigen values
			Mat eigenvalues;
			eigen( harris, eigenvalues ); 
			float eigenvalue1 = eigenvalues.at<float>(0);
			float eigenvalue2 = eigenvalues.at<float>(1);
			harris_eigenvalue1.at<float>(ks, i) = eigenvalue1;
			harris_eigenvalue2.at<float>(ks, i) = eigenvalue2;
		}


		///////////////////////////////////////////////////////////////////////
		// Hessian Matrix
		///////////////////////////////////////////////////////////////////////
		Mat hessian_Ixx, hessian_Ixy, hessian_Iyy;
		GaussianBlur( Ixx, hessian_Ixx, ksize, 0, 0 );
		GaussianBlur( Ixy, hessian_Ixy, ksize, 0, 0 );
		GaussianBlur( Iyy, hessian_Iyy, ksize, 0, 0 );
		// compute the vessel ness
		for( int i=0; i<src.cols; i++ )
		{
			int j = src.rows/2;
			// construct the harris matrix
			Mat hessian( 2, 2, CV_32F );
			hessian.at<float>(0, 0) = hessian_Ixx.at<float>(j, i);
			hessian.at<float>(1, 0) = hessian_Ixy.at<float>(j, i);
			hessian.at<float>(0, 1) = hessian_Ixy.at<float>(j, i);
			hessian.at<float>(1, 1) = hessian_Iyy.at<float>(j, i);
			// calculate the eigen values
			Mat eigenvalues;
			eigen( hessian, eigenvalues ); 
			float eigenvalue1 = eigenvalues.at<float>(0);
			float eigenvalue2 = eigenvalues.at<float>(1);
			if( eigenvalue1<eigenvalue2 ) std::swap( eigenvalue1, eigenvalue2 );
			hessian_eigenvalue1.at<float>(ks, i) = eigenvalue1;
			hessian_eigenvalue2.at<float>(ks, i) = eigenvalue2;
		}


		///////////////////////////////////////////////////////////////////////
		// Optimal Oriented Flux
		///////////////////////////////////////////////////////////////////////
		// Gaussain Filter
		Mat gaussian = getGaussianKernel( s, 0, CV_32F );
		gaussian = gaussian * gaussian.t();
		// Derivatives of Gaussian
		Mat gaussian_dx, gaussian_dy;
		filter2D( gaussian, gaussian_dx, CV_32F, filter_dx );
		filter2D( gaussian, gaussian_dy, CV_32F, filter_dy );
		// Second Derivatives of Gaussian
		Mat gaussian_dxdx, gaussian_dydy, gaussian_dxdy;
		filter2D( gaussian_dx, gaussian_dxdx, CV_32F, filter_dx );
		filter2D( gaussian_dx, gaussian_dxdy, CV_32F, filter_dy );
		filter2D( gaussian_dy, gaussian_dydy, CV_32F, filter_dy );
		// Optimal Oriented Flux Values
		Mat oof_dxdx, oof_dydy, oof_dxdy;
		filter2D( src_gray, oof_dxdx, CV_32F, gaussian_dxdx );
		filter2D( src_gray, oof_dydy, CV_32F, gaussian_dydy );
		filter2D( src_gray, oof_dxdy, CV_32F, gaussian_dxdy );
		// compute the vessel ness
		for( int i=0; i<src.cols; i++ )
		{
			int j = src.rows/2;
			//construct the Q matrix
			Mat Q( 2, 2, CV_32F );
			Q.at<float>(0, 0) = oof_dxdx.at<float>(j, i);
			Q.at<float>(1, 0) = oof_dxdy.at<float>(j, i);
			Q.at<float>(0, 1) = oof_dxdy.at<float>(j, i);
			Q.at<float>(1, 1) = oof_dydy.at<float>(j, i);
			// calculate the eigen values
			Mat eigenvalues;
			eigen( Q, eigenvalues ); 
			float eigenvalue1 = eigenvalues.at<float>(0);
			float eigenvalue2 = eigenvalues.at<float>(1);
			if( eigenvalue1<eigenvalue2 ) std::swap( eigenvalue1, eigenvalue2 );
			oof_eigenvalue1.at<float>(ks, i) = eigenvalue1;
			oof_eigenvalue2.at<float>(ks, i) = eigenvalue2;
		}
	}

	line( src, Point(0, src.rows/2), Point(src.cols-1, src.rows/2), Scalar(255,0,0), 1, CV_AA, 0 );
	imshow( "Image", src);

	Mat background = src_gray.row(src_gray.rows/2);

	////////////////////////////////////////////////////////////////////////////////////////
	// Visualization of Eigenvalues of Harris Detector
	// visualize_veselness( "Harris Eigenvalue 1", background, harris_eigenvalue1, 200 );
	// visualize_veselness( "Harris Eigenvalue 2", background, harris_eigenvalue2, 200 );
	vector<Mat> harris_eigenvalues;
	harris_eigenvalues.push_back( harris_eigenvalue1 );
	harris_eigenvalues.push_back( harris_eigenvalue2 );
	visualize_eigens( "Harris Eigenvalues", background, harris_eigenvalues, 200);

	////////////////////////////////////////////////////////////////////////////////////////
	// Visualization of Eigenvalues of Hessian Matrix
	// visualize_veselness( "Hessian Eigenvalue 1", background, hessian_eigenvalue1, 200);
	// visualize_veselness( "Hessian Eigenvalue 2", background, hessian_eigenvalue2, 200);
	vector<Mat> hessian_eigenvalues;
	hessian_eigenvalues.push_back( hessian_eigenvalue1 );
	hessian_eigenvalues.push_back( hessian_eigenvalue2 );
	visualize_eigens( "Hessian Eigenvalues", background, hessian_eigenvalues, 200);

	////////////////////////////////////////////////////////////////////////////////////////
	// Visualization of Eigenvalues of Otimal Oriented Flux
	// visualize_veselness( "Optimal Oriented Flux Eigenvalue 1", background, oof_eigenvalue1, 200);
	// visualize_veselness( "Optimal Oriented Flux Eigenvalue 2", background, oof_eigenvalue2, 200);
	vector<Mat> oof_eigenvalues;
	oof_eigenvalues.push_back( oof_eigenvalue1 );
	oof_eigenvalues.push_back( oof_eigenvalue2 );
	visualize_eigens( "Optimal Oriented Flux Eigenvalue", background, oof_eigenvalues, 200);
	
	waitKey(0);
	return 0;
}

bool Validation::Hessian_2D(void)
{
	// loading image
	Mat src = imread( "data/2d images/tree smal bright.png");
	if( !src.data ){ 
		cout << "Image not found..." << endl;
		return false; 
	}

	// convert form CV_8UC3 to CV_8U
	Mat src_gray;
	cvtColor( src, src_gray, CV_RGB2GRAY ); 
	
	// Image gradient along x, y direction
	Mat Ix, Iy;
	Sobel( src_gray, Ix, CV_32F, 1, 0, 1 );
	Sobel( src_gray, Iy, CV_32F, 0, 1, 1 );
	// Second order derivative of the image
	Mat Ixx, Ixy, Iyy;
	Sobel( Ix, Ixx, CV_32F, 1, 0, 1 );
	Sobel( Ix, Ixy, CV_32F, 0, 1, 1 );
	Sobel( Iy, Iyy, CV_32F, 0, 1, 1 );

	// derivative fileters
	Mat filter_dx = ( Mat_<float>(1,3) << -0.5, 0, 0.5 );
	Mat filter_dy = ( Mat_<float>(3,1) << -0.5, 0, 0.5 );

	// Calculate the vesselness response
	const int NUM = 1;

	// Eigenvalues result will be stored in these matrix
	Mat hessian_eigenvalue1( src.rows, src.cols, CV_32F );
	Mat hessian_eigenvalue2( src.rows, src.cols, CV_32F );
	Mat hessian_vesselness(  src.rows, src.cols, CV_32F );

	// coresponding sigma
	float sigma = 4.0f;
	// Kernel Size of Gaussian Blur
	int ks = int( ( sigma - 0.35f ) / 0.15f ); 
	if( ks%2==0 ) ks++;
	cv::Size ksize( ks, ks );
	cout << "Kernel Size: " << ks << endl;
	sigma = 0.15f*ks + 0.35f;
	cout << "Sigma: " << sigma << endl;

	static const float beta = 0.20f; 
	static const float c = 70000.0f; 

	///////////////////////////////////////////////////////////////////////
	// Hessian Matrix
	///////////////////////////////////////////////////////////////////////
	Mat hessian_Ixx, hessian_Ixy, hessian_Iyy;
	GaussianBlur( Ixx, hessian_Ixx, ksize, 0, 0 );
	GaussianBlur( Ixy, hessian_Ixy, ksize, 0, 0 );
	GaussianBlur( Iyy, hessian_Iyy, ksize, 0, 0 );

	// normalized them
	float sigma2 = sigma*sigma;
	hessian_Ixx *= sigma2;
	hessian_Ixy *= sigma2;
	hessian_Iyy *= sigma2;

	// compute the vessel ness
	for( int y=0; y<src.rows; y++ ) {
		for( int x=0; x<src.cols; x++ ){
			// construct the harris matrix
			Mat hessian( 2, 2, CV_32F );
			hessian.at<float>(0, 0) = hessian_Ixx.at<float>(y, x);
			hessian.at<float>(1, 0) = hessian_Ixy.at<float>(y, x);
			hessian.at<float>(0, 1) = hessian_Ixy.at<float>(y, x);
			hessian.at<float>(1, 1) = hessian_Iyy.at<float>(y, x);
			// calculate the eigen values
			Mat eigenvalues;
			eigen( hessian, eigenvalues ); 
			float eigenvalue1 = eigenvalues.at<float>(0);
			float eigenvalue2 = eigenvalues.at<float>(1);
			if( abs(eigenvalue1)>abs(eigenvalue2) ) std::swap( eigenvalue1, eigenvalue2 );
			// Now we have |eigenvalue1| < |eigenvalue2| 
			hessian_eigenvalue1.at<float>(y, x) = eigenvalue1;
			hessian_eigenvalue2.at<float>(y, x) = eigenvalue2;
			if( eigenvalue2 > 0 ) {
				hessian_vesselness.at<float>(y, x) = 0;
			} else {
				float RB = eigenvalue1 / eigenvalue2;
				float S = sqrt( eigenvalue1*eigenvalue1 + eigenvalue2*eigenvalue2 );
				hessian_vesselness.at<float>(y, x) = exp( -RB*RB/beta ) * ( 1-exp(-S*S/c) );
			}
		}
	}

	// Vesselness 
	stringstream ss;
	ss << "Vesselness2 " << "ksize=" << ks << " sigma=" << sigma;
	ss << " beta=" << beta << " c=" << c;
	imshow( ss.str(), hessian_vesselness );
	imwrite( ss.str()+".bmp", hessian_vesselness*255 );

	// draw a line
	int row = src.rows*2/3;
	line( src, Point(0, row), Point(src.cols-1, row), Scalar(255,0,0), 1, CV_AA, 0 );
	imshow( "Image", src);
	////////////////////////////////////////////////////////////////////////////////////////
	// Visualization of Eigenvalues of Hessian Matrix
	vector<Mat> eigenvalues;
	eigenvalues.push_back( hessian_eigenvalue1.row( row ) );
	eigenvalues.push_back( hessian_eigenvalue2.row( row ) );
	Mat background = src_gray.row( row );
	visualize_eigens( "Hessian Eigenvalue 1", background, eigenvalues, 200, 1.0f);

	waitKey(0);
	return 0;
}
