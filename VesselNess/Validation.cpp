#include "stdafx.h"
#include "Validation.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include "Data3D.h"
#include "Kernel3D.h"
#include "Image3D.h"
#include "Viewer.h"
#include "ImageProcessing.h" 

namespace Validation{
	namespace Eigenvalues {
		// derivative fileters
		static const Mat dx = ( Mat_<float>(1,3) << -0.5, 0, 0.5 );
		static const Mat dy = ( Mat_<float>(3,1) << -0.5, 0, 0.5 );

		void get_3d_tubes( Data3D<short>& src, vector<float>& radius ) {
			// first of all, construct 3D tubes
			src.reset( Vec3i(500, 100, 100) );

			// radius of vessels
			radius.push_back( 6.0f );
			radius.push_back( 10.0f );
			radius.push_back( 20.0f );

			// center of the vessel
			const Vec3f center[3] = {
				Vec3f(100, 0, 50),
				Vec3f(250, 0, 50),
				Vec3f(400, 0, 50)
			};

			for( int i=0; i<3; i++ ) { 
				for( int y=0; y<src.SY(); y++ ) for( int z=-25; z<25; z++ ) for( int x=-25; x<25; x++ ){
					float dis_center = sqrt( 1.0f*x*x + z*z );
					float ratio = radius[i] + 0.5f - dis_center;
					if( ratio>1.0f ) {
						src.at( int(x+center[i][0]), y, int(z+center[i][2]) ) = MAX_SHORT;
					} else if( ratio>0.0f ) {
						src.at( int(x+center[i][0]), y, int(z+center[i][2]) ) = short( MAX_SHORT * ratio ); 
					}
				}
			}
		}

		void get_2d_tubes( Mat& src, vector<float>& radius ) {
			src = Mat(100, 500, CV_8U, Scalar(0) );

			radius.push_back(6.0f);
			radius.push_back(10.0f);
			radius.push_back(20.0f);

			double centers[3] = { 100.5, 250.0, 399.5 };

			// constructe the image
			for( int i=0; i<3; i++ ) {
				for( int y=0; y<src.rows; y++ ) for( int x=0; x<src.cols; x++ ) {
					if( abs(x-centers[i]) <= radius[i] ) src.at<unsigned char>(y,x) = 255;
				}
			}
		}

		void get_2d_balls( Mat& src, vector<float>& radius ) {
			// construct the image
			src = Mat(100, 500, CV_8U, cv::Scalar(0) );

			// radius of vessels
			radius.push_back( 6.5f );
			radius.push_back( 10.5f );
			radius.push_back( 20.5f );

			// center of the vessel
			const Vec2f center[3] = {
				Vec2f(100.5, 50.5),
				Vec2f(250.5, 50.5),
				Vec2f(400.5, 50.5)
			};

			for( unsigned int i=0; i<radius.size(); i++ ) { 
				for( int y=0; y<src.rows; y++ ) for( int x=0; x<src.cols; x++ ){
					float dx = x - center[i][0];
					float dy = y - center[i][1];
					float dis_center = sqrt( dx*dx + dy*dy );
					float ratio = radius[i] + 0.5f - dis_center;
					if( ratio>1.0f ) {
						src.at<unsigned char>( y,x ) = 255;
					} else if( ratio>0.0f ) {
						src.at<unsigned char>( y,x ) = unsigned char( 255 * ratio ); 
					}
				}
			}
		}

		void get_1d_box( Mat_<double>& src, vector<float>& radius ){
			// Generate a 1d image
			src = Mat_<double>( 500, 1, 0.0 );

			// radius of vessels
			radius.push_back( 6.5f );
			radius.push_back( 10.5f );
			radius.push_back( 20.5f );

			// center of the vessel
			int center[3] = { 100, 250, 400 };

			for( int i=0; i<3; i++ ){ 
				for( int x=center[i]-(int)radius[i]; x<=center[i]+(int)radius[i]; x++ )
					src.at<double>(x) = 255;
			}
		}

		void plot_3d_tubes(void) {
			// first of all, construct 3D tubes
			Data3D<short> src( Vec3i(500, 100, 100) );
			vector<float> radius;
			get_3d_tubes( src, radius );

			// Eigenvalues result will be stored in these matrix
			vector< Mat_<double> > plot_log;
			vector< Mat_<double> > plot_multiply;
			vector< Mat_<double> > plot_square_sum;

			int cy = src.SY()/2;
			int cz = src.SZ()/2;

			for( int i=0; i<3; i++ ) {
				double sigma = radius[i] / sqrt(2.0);
				int ksize = int( 6 * sigma + 1 );
				if( ksize%2==0 ) ksize++;

				Data3D<float> im_blur = src;

				// We could have blur the image here, but the result is just a lot more smoother
				// if I blur the im_dx2, im_dy2
				// bool flag = ImageProcessing::GaussianBlur3D( src, im_blur, ksize, sigma );

				//Normalizing for different scale
				im_blur *= (sigma*sigma);

				static const Kernel3D<float> dx = Kernel3D<float>::dx();
				static const Kernel3D<float> dy = Kernel3D<float>::dy();
				static const Kernel3D<float> dz = Kernel3D<float>::dz();

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

				ImageProcessing::GaussianBlur3D( im_dx2, im_dx2, ksize, sigma );
				ImageProcessing::GaussianBlur3D( im_dy2, im_dy2, ksize, sigma );
				ImageProcessing::GaussianBlur3D( im_dz2, im_dz2, ksize, sigma );
				ImageProcessing::GaussianBlur3D( im_dxdy, im_dxdy, ksize, sigma );
				ImageProcessing::GaussianBlur3D( im_dxdz, im_dxdz, ksize, sigma );
				ImageProcessing::GaussianBlur3D( im_dydz, im_dydz, ksize, sigma );

				Mat_<double> hessian_square_sum( src.SX(), 1);
				Mat_<double> hessian_multiply( src.SX(), 1);
				for( int x=0; x<src.SX(); x++ ) {
					// construct the harris matrix
					Mat hessian( 3, 3, CV_32F );
					hessian.at<float>(0, 0) = im_dx2.at(x, cy, cz);
					hessian.at<float>(1, 1) = im_dy2.at(x, cy, cz);
					hessian.at<float>(2, 2) = im_dz2.at(x, cy, cz);
					hessian.at<float>(1, 0) = hessian.at<float>(0, 1) = im_dxdy.at(x, cy, cz);
					hessian.at<float>(2, 0) = hessian.at<float>(0, 2) = im_dxdz.at(x, cy, cz);
					hessian.at<float>(2, 1) = hessian.at<float>(1, 2) = im_dydz.at(x, cy, cz);

					// calculate the eigen values
					Mat eigenvalues;
					eigen( hessian, eigenvalues ); 
					// order eigenvalues so that |lambda1| < |lambda2| < |lambda3| 
					int i=0, j=1, k=2;
					if( abs(eigenvalues.at<float>(i)) > abs(eigenvalues.at<float>(j)) ) std::swap( i, j );
					if( abs(eigenvalues.at<float>(i)) > abs(eigenvalues.at<float>(k)) ) std::swap( i, k );
					if( abs(eigenvalues.at<float>(j)) > abs(eigenvalues.at<float>(k)) ) std::swap( j, k );
					if( abs(eigenvalues.at<float>(i)) > abs(eigenvalues.at<float>(j)) ) std::swap( i, j );

					hessian_square_sum.at<double>( x ) = eigenvalues.at<float>(k) * eigenvalues.at<float>(k)
						+ eigenvalues.at<float>(j) * eigenvalues.at<float>(j);
					hessian_multiply.at<double>( x) = eigenvalues.at<float>(j) * eigenvalues.at<float>(k);
				}
				plot_square_sum.push_back( hessian_square_sum );
				plot_multiply.push_back( hessian_multiply );
			}

			Mat_<unsigned char> background( src.SX(), 1);
			for( int x=0; x<src.SX(); x++ ) {
				background = 255 - src.at(x, cy, cz) / 65536 * 55;
			}
			VI::OpenCV::plot( "3d_tubes_hessians_square_sum", plot_square_sum );
			VI::OpenCV::plot( "3d_tubes_hessians_multiply", plot_multiply );
		}

		void plot_2d_tubes(void) {
			Mat src;
			vector<float> radius;
			get_2d_tubes( src, radius );

			// Image gradient along x, y direction
			Mat Ix, Iy;
			cv::filter2D( src, Ix, CV_32F, dx);
			cv::filter2D( src, Iy, CV_32F, dy);
			// Second order derivative of the image
			Mat Ixx, Ixy, Iyy;
			cv::filter2D( Ix, Ixx, CV_32F, dx);
			cv::filter2D( Ix, Ixy, CV_32F, dy);
			cv::filter2D( Iy, Iyy, CV_32F, dy);


			// Eigenvalues result will be stored in these matrix
			Mat big_eigen( src.rows, src.cols, CV_32F );

			// get the center row
			int rid = src.rows / 2;

			for( unsigned int i=0; i<radius.size(); i++ ) {
				vector< Mat_<double> > plot_big_eigen;

				// coresponding sigma
				float sigma = radius[i];
				float sigma2 = sigma * sigma;

				///////////////////////////////////////////////////////////////////////
				// Hessian Matrix
				///////////////////////////////////////////////////////////////////////
				Mat hessian_Ixx, hessian_Ixy, hessian_Iyy;
				cv::GaussianBlur( Ixx, hessian_Ixx, cv::Size(), sigma, sigma );
				cv::GaussianBlur( Ixy, hessian_Ixy, cv::Size(), sigma, sigma );
				cv::GaussianBlur( Iyy, hessian_Iyy, cv::Size(), sigma, sigma );

				// for comparison between scales
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
						big_eigen.at<float>(y, x) = abs(eigenvalue1)>abs(eigenvalue2) ? eigenvalue1 : eigenvalue2;
					}
				}
				Mat big_eigen_normalized; 
				cv::normalize( big_eigen,big_eigen_normalized, 0, 255, NORM_MINMAX, CV_8UC1); 
				cv::imshow("The Eigenvalue with a bigger absolute value", big_eigen_normalized);
				cv::imwrite("output/2d_tubes_hessians_bigger_eigenvalue.bmp", big_eigen_normalized);
				waitKey(0); 

				plot_big_eigen.push_back( big_eigen.row( src.rows/2 ).reshape( 0, big_eigen.cols) );
				// Visualization of Eigenvalues of Hessian Matrix
				Mat_<unsigned char> background = src.row( rid ).clone().reshape(0, src.cols);
				background = 255 - background / 5;
				
				stringstream ss;
				ss << "2d_tubes_hessians_" << radius[i]; 
				VI::OpenCV::plot( ss.str(), plot_big_eigen, 100, 0, background );
			}

			// visualzie and save original image
			imshow("Original Image", src);
			imwrite("output/original_image_2d_tubes_20.jpg", src);

			return;
		}

		void plot_2d_ball(void) {
			Mat src;
			vector<float> radius;
			get_2d_balls( src, radius );

			// Eigenvalues result will be stored in these matrix
			vector< Mat_<double> > plot_log; // laplacian of gaussian
			
			vector< Mat_<double> > plot_negative_add;
			vector< Mat_<double> > plot_multiply;
			vector< Mat_<double> > plot_square_sum;
			vector< Mat_<double> > plot_max_abs;
			vector< Mat_<double> > plot_negative_min;
			vector< Mat_<double> > plot_eigenvalues;
			
			// get the center row
			int rid = src.rows / 2;

			for( unsigned int i=1; i<radius.size(); i++ ) {
				// coresponding sigma
				double sigma = radius[i] / sqrt(2.0);
				// Kernel Size of Gaussian Blur
				int ks = int( 6*sigma+ 1 ); 
				if( ks%2==0 ) ks++;

				Mat g = cv::getGaussianKernel( ks, sigma, CV_64F ) * sigma; 
				Mat g2 = g * g.t();

				Mat gx, gy, gxy, gxx, gyy;
				filter2D( g2, gx, CV_64F, dx );
				filter2D( g2, gy, CV_64F, dy );
				filter2D( gx, gxy, CV_64F, dy );
				filter2D( gx, gxx, CV_64F, dx );
				filter2D( gy, gyy, CV_64F, dy );

				Mat log, im_log;
				cv::add( gxx, gyy, log, noArray(), CV_64F );
				filter2D( src, im_log, CV_64F, log );
				// laplacian of gaussian
				plot_log.push_back( im_log.row( rid ).reshape( 0, im_log.cols) );

				Mat fxx, fxy, fyy;
				filter2D( src, fxy, CV_64F, gxy );
				filter2D( src, fxx, CV_64F, gxx );
				filter2D( src, fyy, CV_64F, gyy );
				// compute the vessel ness
				Mat_<double> hessian_negative_add( src.rows, src.cols, CV_32F );
				Mat_<double> hessian_multiply(     src.rows, src.cols, CV_32F );
				Mat_<double> hessian_square_sum(   src.rows, src.cols, CV_32F );
				Mat_<double> hessian_max_abs( src.rows, src.cols, CV_32F );
				Mat_<double> hessian_negative_min( src.rows, src.cols, CV_32F );
				Mat_<double> hessian_eigenvalue1( src.rows, src.cols, CV_32F );
				Mat_<double> hessian_eigenvalue2( src.rows, src.cols, CV_32F );
				
				for( int y=0; y<src.rows; y++ ) for( int x=0; x<src.cols; x++ ){
					// construct the harris matrix
					Mat hessian = (Mat_<double>( 2, 2) << 
						fxx.at<double>(y, x), fxy.at<double>(y, x), 
						fxy.at<double>(y, x), fyy.at<double>(y, x));

					// calculate the eigen values
					Mat eigenvalues;
					eigen( hessian, eigenvalues ); 
					double eigenvalue1 = eigenvalues.at<double>(0);
					double eigenvalue2 = eigenvalues.at<double>(1);
					if( abs(eigenvalue1)>abs(eigenvalue2) ) std::swap( eigenvalue1, eigenvalue2 );
					// Now we have |eigenvalue1| < |eigenvalue2| 
					hessian_negative_add.at<double>(y, x)  = - (eigenvalue1 + eigenvalue2); 
					hessian_multiply.at<double>(y, x) = eigenvalue1 * eigenvalue2;
					hessian_square_sum.at<double>(y, x) = sqrt( eigenvalue1 * eigenvalue1 + eigenvalue2 * eigenvalue2 );
					hessian_max_abs.at<double>(y, x)  = max(abs(eigenvalue1), abs(eigenvalue2) ); 
					hessian_negative_min.at<double>(y, x)  = - min(eigenvalue1, eigenvalue2); 

					if( eigenvalue1 > eigenvalue2 ) {
						hessian_eigenvalue1.at<double>(y, x)  = eigenvalue1; 
						hessian_eigenvalue2.at<double>(y, x)  = eigenvalue2; 
					} else {
						hessian_eigenvalue1.at<double>(y, x)  = eigenvalue2; 
						hessian_eigenvalue2.at<double>(y, x)  = eigenvalue1; 
					}
				}
				plot_negative_add.push_back( hessian_negative_add.row( rid ).reshape( 0, hessian_negative_add.cols) ); 
				plot_multiply.push_back( hessian_multiply.row( rid ).reshape( 0, hessian_multiply.cols) );
				plot_square_sum.push_back( hessian_square_sum.row( rid ).reshape( 0, hessian_square_sum.cols) );
				plot_max_abs.push_back( hessian_max_abs.row( rid ).reshape( 0, hessian_max_abs.cols) ); 
				plot_negative_min.push_back( hessian_negative_min.row( rid ).reshape( 0, hessian_negative_min.cols) ); 
				plot_eigenvalues.push_back( hessian_eigenvalue1.row( rid ).reshape( 0, hessian_eigenvalue1.cols) ); 
				plot_eigenvalues.push_back( hessian_eigenvalue2.row( rid ).reshape( 0, hessian_eigenvalue2.cols) ); 
				break;
			}

			int row = src.rows / 2;

			Mat_<unsigned char> background = src.row( rid ).clone().reshape(0, src.cols) ;
			background = 255 - background / 5;

			// Visualization of Eigenvalues of Hessian Matrix
			VI::OpenCV::plot( "2d_balls_log", plot_log, 100, 0, background );

			VI::OpenCV::plot( "2d_balls_hessians_negative_add", plot_negative_add, 100, 0, background );
			VI::OpenCV::plot( "2d_balls_hessians_multiply", plot_multiply, 100, 0, background );
			VI::OpenCV::plot( "2d_balls_hessians_square_sum", plot_square_sum, 100, 0, background );
			VI::OpenCV::plot( "2d_balls_hessians_negative_min", plot_negative_min, 100, 0, background );
			VI::OpenCV::plot( "2d_balls_hessians_max_abs", plot_max_abs, 100, 0, background );
			VI::OpenCV::plot( "2d_balls_hessians_eigenvalues", plot_eigenvalues, 100, 0, background );
			
			

			imshow("Original Image", src);
			imwrite("output/original_image_2d_balls.jpg", src);

			return;
		}

		void plot_1d_box(void) {
			Mat_<double> im;
			vector<float> radius;
			get_1d_box( im, radius);

			// visualize the 1D image
			vector< Mat_<double> > im_vec;
			im_vec.push_back( im );
			VI::OpenCV::plot( "riginal_image_1d_boxes", im_vec, 100 );

			for( int i=0; i<3; i++ ) {
				// convolution
				vector< Mat_<double> > plot_boxes_gxx;

				// generate the gaussian filter
				float sigma = radius[i];
				int ks = int( sigma * 6 + 1 );
				if( ks%2==0 ) ks++; // make sure kSize is odd
				Mat g = cv::getGaussianKernel( ks, sigma, CV_64F );

				Mat gxx;
				filter2D( g, gxx, CV_64F, dy );
				filter2D( gxx, gxx, CV_64F, dy );
				gxx *= sigma*sigma;

				Mat_<double> boxes_gxx;
				filter2D( im, boxes_gxx, CV_64F, gxx );
				plot_boxes_gxx.push_back( boxes_gxx );
				
				// visualize the 1D image
				stringstream ss; 
				ss << "1d_boxes_2nd_gaussian_" << radius[i]; 
				VI::OpenCV::plot( ss.str(), plot_boxes_gxx, 100 );
			}
		}
	}


	namespace box_func_and_2nd_gaussian{
		static const Mat dx = ( Mat_<float>(1,3) << -0.5, 0, 0.5 );
		static const Mat dy = ( Mat_<float>(3,1) << -0.5, 0, 0.5 );

		int im_width = 1001;
		int box_center = 500;
		int box_radius = 100;

		void plot_different_size(void){
			Mat_<double> im_1d( im_width, 1, 0.0);
			for( int i = box_center-box_radius; i<box_center+box_radius; i++ ) {
				im_1d.at<double>(i) = 1;
			}

			vector< Mat_<double> > plot;
			
			plot.push_back( im_1d );

			double sigmas[3] = {
				1.2*box_radius,
				box_radius,
				0.8*box_radius
			}; 
			for( int i=0; i<3; i++ ) {
				double sigma = sigmas[i];
				int ks = 1000;
				if( ks%2 == 0 ) ks++;
				Mat g = cv::getGaussianKernel( ks, sigma, CV_64F ); 
				g = g * pow( box_radius ,3.0) * sqrt( 2*3.14 );
				Mat gx, gxx;
				filter2D( g,  gx,  CV_64F, dy );
				filter2D( gx, gxx, CV_64F, dy );
				plot.push_back( -gxx );
			}

			VI::OpenCV::plot( "box_2nd_gaussian_different_size", plot );
		}

		void plot_different_pos(void){
			Mat_<double> im_1d( im_width, 1, 0.0);
			for( int i = box_center-box_radius; i<box_center+box_radius; i++ ) {
				im_1d.at<double>(i) = 1;
			}

			vector< Mat_<double> > plot;
			
			plot.push_back( im_1d );

			int offset[3] = { 
				80, 0, -40
			}; 
			for( int i=0; i<3; i++ ) {
				double sigma = box_radius;
				int ks = 1000;
				if( ks%2 == 0 ) ks++;
				Mat g = cv::getGaussianKernel( ks, sigma, CV_64F ); 
				g = g * pow( box_radius ,3.0) * sqrt( 2*3.14 );
				Mat gx, gxx;
				filter2D( g,  gx,  CV_64F, dy );
				filter2D( gx, gxx, CV_64F, dy );
				Mat_<double> gxx_offset( gxx.rows, 1 );

				for( int j=0; j< gxx.rows; j++ ){
					gxx_offset.at<double>(j) = gxx.at<double>( ( j+offset[i]+gxx.rows ) % gxx.rows );
				}
				plot.push_back( -gxx_offset );
			}

			VI::OpenCV::plot( "box_2nd_gaussian_different_pos", plot );
		}

		void plot(void){
			// final result to plot
			vector< Mat_<double> > plot;
			
			double sigma = 10;
			int ks = 1000;
			if( ks%2 == 0 ) ks++;
			Mat g = cv::getGaussianKernel( ks, sigma, CV_64F ); 
			g = g * pow( box_radius ,3.0) * sqrt( 2*3.14 );
			Mat gx, gxx;
			filter2D( g,  gx,  CV_64F, dy );
			filter2D( gx, gxx, CV_64F, dy );
			Mat_<double> gxx_offset( gxx.rows, 1 );

			for( int j=0; j< gxx.rows; j++ ){
				gxx_offset.at<double>(j) = gxx.at<double>( ( j+gxx.rows ) % gxx.rows );
			}
			plot.push_back( gxx_offset );


			VI::OpenCV::plot( "2nd_gaussian", plot );
		}
	}
}








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

// This function calculate the eigenvalues of the hessian matrix of an image
// src_gray: a grey scale 2d image
// sigma: sigma for gaussian blur
Mat_<Vec2f> Hessian_Eigens( Mat src_gray, float sigma ){
	Mat_<Vec2f> eigenvalues( src_gray.size() );

	// Kernel Size of Gaussian Blur
	int ks = int( ( sigma - 0.35f ) / 0.15f ); 
	if( ks%2==0 ) ks++;
	cv::Size ksize( ks, ks );
	cout << "Kernel Size: " << ks << endl;
	sigma = 0.15f*ks + 0.35f;
	cout << "Sigma: " << sigma << endl;

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
	for( int y=0; y<src_gray.rows; y++ ) {
		for( int x=0; x<src_gray.cols; x++ ){
			// construct the harris matrix
			Mat hessian( 2, 2, CV_32F );
			hessian.at<float>(0, 0) = hessian_Ixx.at<float>(y, x);
			hessian.at<float>(1, 0) = hessian_Ixy.at<float>(y, x);
			hessian.at<float>(0, 1) = hessian_Ixy.at<float>(y, x);
			hessian.at<float>(1, 1) = hessian_Iyy.at<float>(y, x);
			// calculate the eigen values
			Mat eigens;
			eigen( hessian, eigens ); 
			float eigenvalue1 = eigens.at<float>(0);
			float eigenvalue2 = eigens.at<float>(1);
			if( abs(eigenvalue1)>abs(eigenvalue2) ) std::swap( eigenvalue1, eigenvalue2 );
			// Now we have |eigenvalue1| < |eigenvalue2| 
			eigenvalues.at<Vec2f>(y, x)[0] = eigenvalue1;
			eigenvalues.at<Vec2f>(y, x)[1] = eigenvalue2;
		}
	}

	return eigenvalues; 
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
	
	// Add some noise to the image
	for( int y=0; y<src_gray.rows; y++ ) for( int x=0; x<src_gray.cols; x++ ){
		if( rand()%100 < 20 ) src_gray.at<unsigned char>(y,x) = rand()%255;
	}
	imshow("Oringal With Noise", src_gray);
	imwrite("output/tree_with_noise.jpg", src_gray);
	
	// parameters
	static const float beta = 0.280f; 
	static const float c = 500000.0f; 

	Mat vesselness( src.rows, src.cols, CV_32F );
	Mat_<Vec2f> eigenvalues; 
	for( float sigma = 1.2f; sigma<5.0f; sigma+=0.3f ) {
		eigenvalues = Hessian_Eigens( src_gray, sigma );
		// compute the vessel ness
		for( int y=sigma; y<src.rows-sigma; y++ ) for( int x=sigma; x<src.cols-sigma; x++ ){
			float vn = 0; // vesselness on this point 
			// calculate the eigen values
			const float& eigenvalue1 = eigenvalues.at<Vec2f>(y,x)[0];
			const float& eigenvalue2 = eigenvalues.at<Vec2f>(y,x)[1];
			if( eigenvalue2 < 0 ) {
				float RB = eigenvalue1 / eigenvalue2;
				float S = sqrt( eigenvalue1*eigenvalue1 + eigenvalue2*eigenvalue2 );
				vn = exp( -RB*RB/beta ) * ( 1-exp(-S*S/c) );
			}
			vesselness.at<float>(y,x) = max( vn, vesselness.at<float>(y,x) ); 
		}
	}

	Mat vesselness_char;
	cv::normalize( vesselness, vesselness_char, 255, 0, CV_MINMAX, CV_8UC3 ); 
	imshow( "vesselness_2d_trees.jpg", vesselness_char);
	cv::imwrite( "output/vesselness_2d_trees.jpg", vesselness_char );
	waitKey(); 
	
	return 0; 
	// The following is just for some clumssy visulization of 
	// the eigenvalues of the hessian matrix on a scan line
	// uncommmand the return 0 above if you want to see. 

	// Eigenvalues result will be stored in these matrix
	Mat hessian_eigenvalue1( src.rows, src.cols, CV_32F );
	Mat hessian_eigenvalue2( src.rows, src.cols, CV_32F );
	for( int y=0; y<src.rows; y++ ) for( int x=0; x<src.cols; x++ ){
		// calculate the eigen values
		hessian_eigenvalue1.at<float>(y, x) = eigenvalues.at<Vec2f>(y,x)[0];
		hessian_eigenvalue2.at<float>(y, x) = eigenvalues.at<Vec2f>(y,x)[1];
	}
	// Draw a scan line on the original data
	int row = src.rows*2/3;
	line( src, Point(0, row), Point(src.cols-1, row), Scalar(255,0,0), 1, CV_AA, 0 );
	imshow( "Image", src);
	////////////////////////////////////////////////////////////////////////////////////////
	// Visualization of Eigenvalues of Hessian Matrix
	vector<Mat> eigenvalues_plot;
	eigenvalues_plot.push_back( hessian_eigenvalue1.row( row ) );
	eigenvalues_plot.push_back( hessian_eigenvalue2.row( row ) );
	Mat background = src_gray.row( row );
	visualize_eigens( "Hessian Eigenvalue 1", background, eigenvalues_plot, 200, 1.0f);

	waitKey(0);
	return 0;
}
