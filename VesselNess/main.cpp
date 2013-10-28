// VesselNess.cpp : Defines the entry point for the console application.
//
#define _CRT_SECURE_NO_DEPRECATE

#include "stdafx.h"
#include "VesselDetector.h"
#include "Validation.h"
#include "Kernel3D.h"
#include "Viewer.h"
#include "RingsDeduction.h"
#include "Image3D.h"
#include "ImageProcessing.h"
#include "Vesselness.h"

Data3D<Vesselness_All> vn_all;
Image3D<short> image_data;

class Preset {
public:
	Preset(){};
	Preset(const string& file, Vec3i& size = Vec3i(0,0,0) ) : file(file), size(size){ };
	string file;
	Vec3i size;
};


// some sample functions
void plot_histogram_in_matlab(void) ;
void plot_2d_hessian(void);
void plot_1d_hessian(void);

int main(int argc, char* argv[])
{
	// plot_1d_hessian(); return 0;
	plot_2d_hessian(); return 0;
	plot_histogram_in_matlab(); return 0;
	
	bool flag = false;
	Preset presets[30];
	presets[0] = Preset("vessel3d", Vec3i(585, 525, 892));
	presets[1] = Preset("roi1.data", Vec3i(45, 70, 15));
	presets[2] = Preset("roi2.data", Vec3i(111, 124, 43));
	presets[3] = Preset("roi3", Vec3i(70, 123, 35));
	presets[4] = Preset("roi4", Vec3i(29, 30, 56));
	presets[7] = Preset("roi7", Vec3i(65, 46, 55));
	presets[8] = Preset("roi8.data", Vec3i(102, 96, 34));
	presets[9] = Preset("roi9", Vec3i(129, 135, 97));
	presets[10] = Preset("roi10", Vec3i(51, 39, 38));
	presets[11] = Preset("roi11.data", Vec3i(116, 151, 166));
	presets[12] = Preset("vessel3d.rd.k=19.data", Vec3i(585, 525, 892));
	presets[13] = Preset("roi13.data", Vec3i(238, 223, 481) );
	// presets[14] = Preset("roi14.data" );
	presets[15] = Preset("roi15.data" );
	presets[16] = Preset("roi16.data" );
	presets[17] = Preset("roi17.data" );
	
	const Preset& ps = presets[17];
	string data_name = "roi17.partial";
	// string data_name = "temp";


	


	if( bool isConstructTube = false ) {
		Validation::construct_tube2( image_data );
		//image_data.show("", 50);
		//return 0;
	} 

	/////////////////////////////////////////////////////////////////////////////////////
	//// For tuning parameters
	//flag = image_data.loadROI( ps.file );
	//if( !flag ) return 0;
	//static const int NUM_S = 4;
	//float sigma[] = { 0.3f, 0.5f, 0.7f, 1.5f };
	//// float sigma[] = { 0.5f, 1.5f, 2.5f };
	//int margin = (int) sigma[NUM_S-1]*6 + 20;
	//Vec3i size = image_data.getROI().get_size() - 2* Vec3i(margin, margin, margin);
	//size[0] *= NUM_S;
	//Data3D<Vesselness_All> vn_temp(size);
	//for( int i=0; i<NUM_S; i++ ){ 
	//	VD::compute_vesselness( image_data.getROI(), vn_all, sigma[i], 3.1f, 100.0f );
	//	vn_all.remove_margin( Vec3i(margin+30,margin,margin), Vec3i(margin,margin,margin) );
	//	int x,y,z;
	//	for(z=0;z<vn_all.SZ();z++ ) for(y=0;y<vn_all.SY();y++ ) for(x=0;x<vn_all.SX();x++ ) {
	//		vn_temp.at( x + i*vn_all.SX(), y, z ) = vn_all.at( x, y, z );
	//	}
	//}
	//Viewer::MIP::Multi_Channels( vn_temp, data_name+".vesselness" );
	//return 0;

	if( bool compute = true ) {
		int margin = VD::compute_vesselness( image_data.getROI(), vn_all, 
			// /*sigma from*/ 0.3f, /*sigma to*/ 3.5f, /*sigma step*/ 0.1f );
		    /*sigma from*/ 0.5f, /*sigma to*/ 3.5f, /*sigma step*/ 0.1f );
		
		flag = vn_all.remove_margin( margin ); if( !flag ) return 0;
		vn_all.save( data_name+".vesselness" );
		
		flag = image_data.getROI().remove_margin( margin ); if( !flag ) return 0;
		image_data.getROI().save( data_name + ".data" );

		Viewer::MIP::Multi_Channels( vn_all, data_name + ".vesselness" );
		Viewer::MIP::Single_Channel( image_data.getROI(), data_name + ".data" );
		return 0;

	} else {

		flag = vn_all.load( data_name+".vesselness" );
		if( !flag ) return 0;

		flag = image_data.load( data_name+".data" );
		if( !flag ) return 0;
	}
	

	Data3D<Vesselness_Sig> vn_sig( vn_all );
	//vn_sig.save( data_name+".float5.vesselness" );
	//Viewer::MIP::Multi_Channels( vn_sig, data_name+".float5.vesselness" );
	//Viewer::OpenGL::show_dir( vn_sig );
	//return 0;

	if( bool minimum_spinning_tree = false ) { 
		
		Data3D<Vesselness_Sig> res_mst;
		IP::edge_tracing_mst( vn_all, res_mst, 0.55f, 0.065f  );
		// res_dt.save( data_name + ".dir_tracing.vesselness" );
		Viewer::MIP::Multi_Channels( res_mst, data_name + ".mst" );
		return 0;
	} else {
		Data3D<Vesselness_Sig> res_nms; // result of non-maximum suppression
		IP::non_max_suppress( vn_all, res_nms );
		res_nms.save( data_name + ".non_max_suppress.vesselness" );
		// Viewer::MIP::Multi_Channels( res_nms, data_name + ".non_max_suppress" ); // Visualization using MIP
		//return 0;

		Data3D<Vesselness_Sig> res_rns_et;
		IP::edge_tracing( res_nms, res_rns_et, 0.55f, 0.055f );
		res_rns_et.save( data_name + ".edge_tracing.vesselness" );
		Viewer::MIP::Multi_Channels( res_rns_et, data_name + ".edge_tracing"); // Visualization using MIP
	}

	

	return 0;

	///////////////////////////////////////////////////////////////
	// Ring Recuction by slice
	////////////////////////////////////////////////////////////////
	////Get a clice of data
	//Mat im_slice = image_data.getByZ( 55 );
	//Mat im_no_ring;
	//for( int i=0; i<5; i++ ) {
	//	Validation::Rings_Reduction_Polar_Coordinates( im_slice, im_no_ring, 9 );
	//	im_slice = im_no_ring;
	//}


	return 0;
}


void plot_histogram_in_matlab(void) {
	cout << "Plotting Histogram of data in Matlab. " << endl;
	// loading data
	Image3D<short> im_data;
	bool flag = im_data.loadData( "data/vessel3d.data", Vec3i(585,525,200), true, true );
	if( !flag ) return;
	// calculating histogram
	Mat_<double> hist, range;
	IP::histogram( im_data, range, hist, 1024 );
	VI::Matlab::plot( range, hist );
}


void plot_3d_hessian(void) {
	// first of all, construct 3D tubes

}

void plot_2d_hessian(void) {
	// loading image
	Mat src = imread( "data/images/vessels_2d.bmp");
	if( !src.data ){ cout << "Image not found..." << endl; return; }

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

	float sigmas[3] = { 4.0f, 6.0f, 8.0f};


	vector< Mat_<double> > eigenvalues;

	for( int i=0; i<3; i++ ) {
		// coresponding sigma
		float sigma = sigmas[i];
		// Kernel Size of Gaussian Blur
		int ks = int( ( sigma - 0.35f ) / 0.15f ); 
		if( ks%2==0 ) ks++;
		cv::Size ksize( ks, ks );

		static const float beta = 0.20f; 
		static const float c = 70000.0f; 

		///////////////////////////////////////////////////////////////////////
		// Hessian Matrix
		///////////////////////////////////////////////////////////////////////
		Mat hessian_Ixx, hessian_Ixy, hessian_Iyy;
		GaussianBlur( Ixx, hessian_Ixx, ksize, sigma, sigma );
		GaussianBlur( Ixy, hessian_Ixy, ksize, sigma, sigma );
		GaussianBlur( Iyy, hessian_Iyy, ksize, sigma, sigma );

		// normalized them
		hessian_Ixx *= sigma * sigma;
		hessian_Ixy *= sigma * sigma;
		hessian_Iyy *= sigma * sigma;

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

		eigenvalues.push_back( hessian_eigenvalue2.row( src.rows/2 ).reshape( 0, hessian_eigenvalue1.cols) );
	}
	
	
	int row = src.rows / 2;
	// draw a line on the oringinal image
	line( src, Point(0, row), Point(src.cols-1, row), Scalar(255,0,0), 1, CV_AA, 0 );
	imshow( "Image", src);

	Mat_<unsigned char> background = src_gray.row( row );
	for( int i=0; i<background.cols; i++ ) {
		background.at<unsigned char>(0,i) = 255 - background.at<unsigned char>(0,i)/5;
	}
	
	// Visualization of Eigenvalues of Hessian Matrix
	stringstream plot_name;
	plot_name << "Hessian_Eigenvalue_2D_sigma_2_4_8_16";
	VI::OpenCV::plot( plot_name.str(), eigenvalues, 800, 0, background );

	waitKey(0);
	return;
}


void plot_1d_hessian(void) {
	// Generate a 1d image
	Mat_<double> im( 400, 1, 0.0 );
	for( int i=90; i<101; i++ ) im.at<double>(i) = 10.0;
	for( int i=260; i<281; i++ ) im.at<double>(i) = 10.0;
	
	// visualize the 1D image
	vector< Mat_<double> > im_vec;
	im_vec.push_back( im );
	VI::OpenCV::plot( "im_1d", im_vec, 100, 600 );

	double sigmas[3] = { 5.5, 10.5 };

	vector< Mat_<double> > res_vec;
	for( int i=0; i<2; i++ ) {
		// generate the gaussian filter
		double sigma = sigmas[i];
		int kSize = int( sigma * 6 + 1 );
		if( kSize%2==0 ) kSize++; // make sure kSize is odd
		Mat gaussian = cv::getGaussianKernel( kSize, sigma, CV_64F );
		Mat filter_dy = ( Mat_<float>(3,1) << -0.5, 0, 0.5 );
		Mat gaussian_2nd;
		filter2D( gaussian, gaussian_2nd, CV_64F, filter_dy );
		filter2D( gaussian_2nd, gaussian_2nd, CV_64F, filter_dy );
		gaussian_2nd *= sigma*sigma;

		// convolution
		Mat_<double> response( im.rows, 1, 0.0);
		for( int i=0; i<im.rows; i++ ) {
			for( int j=0; j<gaussian_2nd.rows; j++ ) {
				int im_pos = i-gaussian_2nd.rows/2+j;
				if( im_pos >= 0 && im_pos<im.rows ) {
					response.at<double>(i) += gaussian_2nd.at<double>(j) * im.at<double>(im_pos);
				}
			}
		}
		res_vec.push_back( response );
	}
	// visualize the 1D image
	
	VI::OpenCV::plot( "hessian_1d_sigma2", res_vec, 100, 600 );

	waitKey(0);
}