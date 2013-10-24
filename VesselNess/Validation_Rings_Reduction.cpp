#include "stdafx.h"
#include "Validation.h"
#include "Image3D.h"
#include "ImageProcessing.h"

#define _USE_MATH_DEFINES
#include <math.h>

// plot the values
bool plot( Mat data ){
	smart_return_false( data.type()==CV_64F, "Can only plot data type double." );
	smart_return_false( data.rows<=7, "Can only support at most 7 rows of data.");
	
	// Different Color for defferent row of data
	Scalar colors[7] = {
		Scalar(255,255,255),
		Scalar(255,0,0),
		Scalar(0,255,0),
		Scalar(0,0,255),
		Scalar(255,255,0),
		Scalar(255,0,255),
		Scalar(0,255,255)
	};

	// find the maximum and minimun value in the data
	// find the maximum and minimum vesselness
	double minVal, maxVal;
	minMaxLoc( data, &minVal, &maxVal, NULL, NULL);
	// the difference between the maximum value and the minimum value
	double maxMinDiff = maxVal - minVal;

	// magin of the figure
	static const float margin = 0.05f;
	// the height of the figure
	int im_height = int( 0.618*data.cols );
	// constuct the figure
	Mat im( im_height, data.cols, CV_8UC3, Scalar(0,0,0) );
	// plot the data
	for( int i=0; i<data.rows; i++ ){
		for( int j=1; j<data.cols; j++ ) {
			double v1 = data.at<double>(i, j-1);
			double v2 = data.at<double>(i, j);
			// get the end point of the line
			Point p1, p2;
			p1.x = j-1;
			p1.y = int( im_height * ( margin + (1-2*margin)*(1.0 - (v1-minVal)/maxMinDiff ) ) );
			p2.x = j;
			p2.y = int( im_height * ( margin + (1-2*margin)*(1.0 - (v2-minVal)/maxMinDiff ) ) );
			// draw the line
			line( im, p1, p2, colors[i], 1, CV_AA );
		}
	}
	// draw a line for y = 0
	int y0 = int( im_height * ( margin + (1-2*margin)*(1.0 - (0-minVal)/maxMinDiff ) ) );
	line( im, Point(0, y0), Point(im.cols-1, y0), Scalar(55,55,55), 1, CV_AA );

	// Show the result
	stringstream ss;
	static int i = 0;
	ss << "Figure " << ++i << ".bmp";
	imshow( ss.str(), im );
	imwrite( ss.str(), im );
	return true;
}

void show_n_save( const string& name, Mat im ){
	if( im.type() != CV_16S ){
		cout << "Error: Data Type Not Supported. " << endl;
		return;
	}
	im = im/255 + 127;
	im.convertTo( im, CV_8U );
	imshow( name, im );
	imwrite( name+".bmp", im );
}

template<typename T>
T interpolate(const Mat& mat, double x, double y){
	T result = 0;
	// interpolation
	double ceil_x = ceil(x);
	double floor_x= floor(x);
	double ceil_y = ceil(y);
	double floor_y = floor(y);
	
	
	// if the value happens to be integer
	if( ceil_y==floor_y ) ceil_y = ceil_y + 1; 
	if( ceil_x==floor_x ) ceil_x = ceil_x + 1; 
	// if ceil_y exceed the maximum value, round it up to the very begin line
	if( ceil_y>=mat.rows ) ceil_y -= mat.rows;
	if( floor_y>=mat.rows ) floor_y -= mat.rows;
	// 1)
	double ratio = (ceil_y - y) * (ceil_x - x);
	if( ratio>1e-9) result += T( mat.at<T>( (int) floor_y, (int) floor_x ) * ratio );
	// 2)
	ratio = (y - floor_y) * (ceil_x - x);
	if( ratio>1e-9) result += T( mat.at<T>( (int) ceil_y,  (int) floor_x ) * ratio );
	// 3) 
	ratio = (ceil_y - y) * (x - floor_x);
	if( ratio>1e-9) result += T( mat.at<T>( (int) floor_y, (int) ceil_x )  * ratio );
	// 4)
	ratio = (y - floor_y) * (x - floor_x);
	if( ratio>1e-9) result += T( mat.at<T>( (int) ceil_y,  (int) ceil_x )  * ratio );
	return result;
}

bool Validation::Rings_Reduction_Polar_Coordinates( const Mat& im, Mat& dst, int wsize ){
	smart_return_false( im.type()==CV_16S, "Only support CV_32S" );
	// Center of the ring. Always fixed. 
	// Can be computed using RANSAC, but hard-coded the center 
	// for idea validation only. 
	static const int center_y = 270;
	static const int center_x = 234;

	static int count1 = 0;
	stringstream ss1;
	ss1 << "Image Cartecian Input " << ++count1;
	ImageProcessing::imNormShow( ss1.str(), im.clone() );

	// Define ROI, which is a doughnut
	// The radius will be from [Rmin, Rmax-1]
	int Rmin = 0, Rmax = 230; 
	double dTheta = 1.0f / Rmax; 

	// Transform to Polar coordinates
	Mat im_p( int(2*M_PI*Rmax + 1), Rmax-Rmin, CV_16S, Scalar(0) );
	for( int radius=0; radius<im_p.cols; radius++ ) {
		int r = radius + Rmin;
		for( int row=0; row<im_p.rows; row++ ) {
			double theta = row * dTheta;
			double x = r * cos( theta );
			double y = r * sin( theta );
			// interpolation
			im_p.at<short>(row, radius) = interpolate<short>( im, center_x+x, center_y+y );
		}
	}
	// ImageProcessing::imNormShow( "Image Polar", im_p.clone() );
	
	// Ring Reduction in Polar Coordinates
	// artifacts template and 
	Mat a(1, im_p.cols, CV_64F, Scalar(0) );
	// the corresponding Kn_value for the tempalte a above
	Mat a_kn(1, im_p.cols, CV_32S, Scalar(0) );
	// sorted value for median filter
	Mat An( im_p.rows, wsize, CV_64F);
	for( int x=0; x<im_p.cols-wsize; x++ ) {
		int Kn = 0;
		for( int y=0; y<im_p.rows; y++ ) {
			double sum = 0.0;
			for( int i=0; i<wsize; i++ ) {
				sum += im_p.at<short>(y, x+i);
			}
			double mean = sum / wsize;

			//// the variance of the window
			//double variance = 0.0; 
			//for( int i=0; i<wsize; i++ ) {
			//	double diff = 1.0*im_p.at<short>(y, x+i)-mean;
			//	variance += diff*diff;
			//}
			//variance /= wsize;
			//// threshold for signal variance		
			//if( variance > 2e5 ) continue;
			
			// compute An and sort it
			for( int i=0; i<wsize; i++ ){
				// the difference between the image intensity and the mean value
				// with the window
				double diff = 1.0*im_p.at<short>(y, x+i)-mean;
				// inseart and sort An at column i
				int j;
				for( j=Kn-1; j>=0; j-- ){
					if( An.at<double>(j ,i) > diff ){
						An.at<double>(j+1 ,i) = An.at<double>(j, i);
					} else {
						break;
					}
				}
				An.at<double>(j+1 ,i) = diff;	
			}
			Kn++; // Assert: Kn is no bigger than im_p.rows-1
		}

		// Now we have the mean values at
		for( int i=0; i<wsize; i++ ){
			if( a_kn.at<int>(0, x+i) < Kn ){
				a_kn.at<int>(0, x+i) = Kn;
				a.at<double>(0, x+i) = An.at<double>(Kn/2, i);
			}
		}
	}

	// plot the value
	plot(a);

	Mat im_p_result = im_p.clone();
	for( int x=0; x<im_p_result.cols-wsize; x++ ) {
		for( int y=0; y<im_p_result.rows; y++ ) {
			im_p_result.at<short>(y, x) -= short( a.at<double>(0, x) );
		}
	}

	// ImageProcessing::imNormShow( "Rings Reduction", im_p_result.clone() );
	
	// dst = im.clone();
	dst = Mat::zeros( im.rows, im.cols, im.type() );
	// Transform to Cartecian coordinates
	for( int x=0; x<=Rmax-1; x++ ) {
		int y_min = max(0, Rmin*Rmin-x*x);
		y_min = int( ceil( sqrt( 1.0*y_min ) ) );
		int y_max = (Rmax-1)*(Rmax-1)-x*x; 
		y_max = int( floor( sqrt( 1.0*y_max ) ) );
		// Assert: both y_min and y_max >= 0
		for( int y=y_min; y<=y_max; y++ ) { 
			// rasius of the ring
			double radius = sqrt( 1.0 * x * x + y * y ) - Rmin;
			// angle of the ring
			double angle = ( x==0 ) ? 0.5*M_PI : atan( 1.0*y/x );
			// interpolation for all for quandrant
			// Quandrant 1
			// if( x==0 && x==1 ) continue; 
			dst.at<short>(center_y+y, center_x+x) = interpolate<short>(im_p_result, radius, angle / dTheta );
			// Quandrant 2
			dst.at<short>(center_y+x, center_x-y) = interpolate<short>(im_p_result, radius, (angle+0.5*M_PI) / dTheta );
			// Quandrant 3
			dst.at<short>(center_y-y, center_x-x) = interpolate<short>(im_p_result, radius, (angle+M_PI) / dTheta );
			// Quandrant 4
			dst.at<short>(center_y-x, center_x+y) = interpolate<short>(im_p_result, radius, (angle+1.5*M_PI) / dTheta );
		}
	}

	static int count = 0;
	stringstream ss;
	ss << "Image Cartecian Result " << ++count;
	ImageProcessing::imNormShow( ss.str(), dst.clone() );
	
	return true;
}


bool Validation::Rings_Reduction_Cartecian_Coordinates(const Mat& src, Mat& dst){
	ImageProcessing::imNormShow( "Source Image", src );
	// Center of the ring. Always fixed. 
	// Can be computed using RANSAC, but hard-coded the center 
	// for idea validation only. 
	static const int center_y = 270;
	static const int center_x = 234;

	// Use a 9*9 box mean filter
	boxFilter( src, dst, -1, cv::Size(9,9) );
	ImageProcessing::imNormShow( "Source Image Result", dst );

	Mat diff = src - dst;
	diff.convertTo( diff, CV_32F );
	multiply( diff, diff, diff );
	ImageProcessing::imNormShow( "Src - Dst", diff );

	MatIterator_<float> it;
	for( it=diff.begin<float>(); it<diff.end<float>(); it++ ) {
		(*it) = ( (*it)>2e6 ) ? 1.0f : 0.0f;
	}
	imshow( "Src - Dst, threshold", diff );

	waitKey(0);

	// TODO: This function is not finish yet.
	// A lot more need to be added here


	return true;
}