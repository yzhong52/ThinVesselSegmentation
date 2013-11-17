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
#include "GLViewer.h"
#include "GLViewerExt.h"
#include "MinSpanTree.h"


Data3D<Vesselness_All> vn_all;
Image3D<short> image_data;

void plot_histogram_in_matlab(void) {
	smart_assert( 0, "Deprecated!" );
	//cout << "Plotting Histogram of data in Matlab. " << endl;
	//// loading data
	//Image3D<short> im_data;
	//bool flag = im_data.loadData( "data/vessel3d.data", Vec3i(585,525,200), true, true );
	//if( !flag ) return;
	//// calculating histogram
	//Mat_<double> hist, range;
	//IP::histogram( im_data, range, hist, 1024 );
	//MLVier::plot( range, hist );
}


bool set_roi_for_vesselness( void ) {
	bool flag; 
	// load the image data
	Image3D<short> image_data;
	flag = image_data.load( "data/roi16.partial.data" ); 
	if( !flag ) return 0;
	// load roi the image data
	image_data.loadROI( "data/roi16.partial.roi.data" );
	VI::MIP::Single_Channel( image_data.getROI(), "data/roi16.partial.roi.data" );
	// load vesselness
	Image3D<Vesselness_Sig> vn_sig1;
	bool flag3 = vn_sig1.load( "data/roi16.partial.float5.vesselness" );
	if( flag3 == 0 ) return 0;
	// set roi for vesselness
	vn_sig1.setROI( image_data.get_roi_corner_0(), image_data.get_roi_corner_1() );
	// save roi for vesselness
	vn_sig1.saveROI( "data/roi16.partial.roi.float5.vesselness" );
	VI::MIP::Multi_Channels( vn_sig1.getROI(), "data/roi16.partial.roi.float5.vesselness" );
	return true;
}

void compute_vesselness(void){
	string data_name = "roi15";

	bool flag = image_data.loadROI( "data/"+data_name+".data" );
	if( !flag ) return;
	
	Data3D<Vesselness> vn;
	VD::compute_vesselness( image_data.getROI(), vn, 
		/*sigma from*/ 0.5f,
		/*sigma to*/ 3.5f,
		/*sigma step*/ 1.5f );
	
	string vn_name = "output/" + data_name + ".float4.vesselness";
	vn.save( vn_name );
	Viewer::MIP::Multi_Channels( vn, vn_name );

} 






int main(int argc, char* argv[])
{
	//waitKey();
	//return 0;

	struct Preset {
		Preset(){}
		Preset(const string& file, Vec3i& size = Vec3i(0,0,0) ) : file(file), size(size){ };
		string file;
		Vec3i size;
	};

	
	bool flag = false;
	Preset presets[30];
	presets[0] = Preset("data/vessel3d.data", Vec3i(585, 525, 892));
	presets[10] = Preset("roi10", Vec3i(51, 39, 38));
	presets[11] = Preset("roi11.data", Vec3i(116, 151, 166));
	presets[12] = Preset("vessel3d.rd.k=19.data", Vec3i(585, 525, 892));
	presets[13] = Preset("roi13.data", Vec3i(238, 223, 481) );
	// presets[14] = Preset("roi14.data" );
	presets[15] = Preset("data/roi15.data" );
	presets[16] = Preset("roi16.data" );
	presets[17] = Preset("roi17.data" );
	
	const Preset& ps = presets[12];
	string data_name = "temp";


	bool falg = image_data.load( "data/roi16.partial.data" );
	if( !falg ) return 0;

	Image3D<unsigned char> image_data_uchar;
	IP::normalize( image_data.getROI(), short(255) );
	image_data.getROI().convertTo( image_data_uchar );
	
	// Computer Min Span Tree
	Graph< MST::Edge_Ext, MST::LineSegment > tree;
	MinSpanTree::build_tree_xuefeng( "data/roi16.partial.linedata.txt", tree, 150 );
	// Visualize Min Span Tree on Max Intensity Projection
	GLViewerExt::draw_min_span_tree_init( tree );
	GLViewerExt::save_video_int( "output/video.avi", 20, 18 );
	GLViewer::MIP( image_data_uchar.getROI().getMat().data, 
		image_data_uchar.SX(),
		image_data_uchar.SY(),
		image_data_uchar.SZ(), 
		GLViewerExt::draw_min_span_tree, // drawing min span tree
		GLViewerExt::save_video );       // saving video

	return 0;

	// image_data.loadData( presets[0].file, presets[0].size );
	//image_data.setROI();
	/*image_data.loadData( "data/roi16.partial.partial.data", Vec3i(111,44,111), false );
	
	Image3D<unsigned char> image_data_uchar;
	IP::normalize( image_data, short(255) );
	image_data.convertTo( image_data_uchar );
	image_data_uchar.show();
	image_data_uchar.save( "output/roi16.partial.partial.uchar.data" );
	VI::MIP::Single_Channel( image_data_uchar, "output/temp" );
*/

	compute_vesselness();
	
	return 0;


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
		flag = image_data.loadROI("data/roi15.data");
		if( !flag ) return 0;

		VD::compute_vesselness( image_data.getROI(), vn_all, 
			// /*sigma from*/ 0.3f, /*sigma to*/ 3.5f, /*sigma step*/ 0.1f );
		    /*sigma from*/ 0.5f, /*sigma to*/ 3.5f, /*sigma step*/ 0.5f );
	
		vn_all.save( data_name+".vesselness" );
		
		Viewer::MIP::Multi_Channels( vn_all, data_name + ".vesselness" );
		// Viewer::MIP::Single_Channel( image_data.getROI(), data_name + ".data" );
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

	////////////////////////////////////////////////////////////////
	// Plotting About Eigenvalues
	////////////////////////////////////////////////////////////////
	//// Visualization of the Eigenvalues
	// Validation::Eigenvalues::plot_1d_box();
	// Validation::Eigenvalues::plot_2d_tubes();
	// Validation::Eigenvalues::plot_2d_ball();
	// Validation::Eigenvalues::plot_3d_tubes();
	// Validation::BallFittingModels::cylinder();
	
	//// Draw Second Derivative of Gaussian on top of the Box function 
	// Validation::box_func_and_2nd_gaussian::plot_different_size();
	// Validation::box_func_and_2nd_gaussian::plot_different_pos();

	return 0;
}
