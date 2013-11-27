// VesselNess.cpp : Defines the entry point for the console application.
//
#define _CRT_SECURE_NO_DEPRECATE

#include "stdafx.h"
#include "VesselDetector.h"
#include "Viewer.h"
#include "RingsDeduction.h"
#include "Image3D.h"
#include "ImageProcessing.h"
#include "Vesselness.h"

#include "MinSpanTree.h"
#include "MinSpanTreeWrapper.h"

// OpenGL Viewer With Maximum Intensity Projection
#include "GLViewer.h"
#include "VideoSaver.h"
#include "GLViwerWrapper.h"

#include "CenterLine.h"
#include "Volumn.h"


void compute_vesselness( string dataname = "vessel3d.rd.19",
	float sigma_from = 0.5, float sigma_to = 45.0f, float sigma_step = 1.0 )
{
	// laoding data
	Data3D<short> im_short;
	bool falg = im_short.load( "data/"+dataname+".data" );
	if(!falg) return;
	
	stringstream vesselness_name;
	vesselness_name << "output/";
	vesselness_name << dataname;
	vesselness_name << ".sigma_to" << sigma_to;

	stringstream vesselness_log;
	vesselness_log << " sigma from " << sigma_from;
	vesselness_log << " to "   << sigma_to;
	vesselness_log << " with setp " << sigma_step;

	// compute vesselness
	Data3D<Vesselness_All> vn_all;
	vn_all.resize( im_short.get_size() );
	VD::compute_vesselness( im_short, vn_all, sigma_from, sigma_to, sigma_step);
	vn_all.save( vesselness_name.str()+".vn_all", vesselness_log.str() );

	Data3D<Vesselness_Sig> vn_sig( vn_all );
	vn_sig.save( vesselness_name.str()+".vn_sig", vesselness_log.str() );

	Data3D<float> vn_float; 
	vn_sig.copyDimTo( vn_float, 0 );
	vn_float.save( vesselness_name.str()+".vn_float", vesselness_log.str() );
	
	GLViewer::MIP( vn_float );
}

void compute_min_span_tree( string data_name = "roi16.partial" ) {
	Data3D<short> im_short;

	im_short.load( "data/" +data_name+ ".data" );
	
	Image3D<unsigned char> im_uchar;
	IP::normalize( im_short, short(255) );
	im_short.convertTo( im_uchar );
	
	vector<GLViewer::Object*> objs;
	GLViewerExt::Volumn vObj( im_uchar.getMat().data, 
		im_uchar.SX(), im_uchar.SY(), im_uchar.SZ() );
	objs.push_back( &vObj );


	// Computer Min Span Tree
	Graph< MST::Edge_Ext, MST::LineSegment > tree;
	MinSpanTree::build_tree_xuefeng( "data/" +data_name+ ".linedata.txt", tree, 150 );

	GLViewer::CenterLine<MST::Edge_Ext, MST::LineSegment> cObj( tree );
	objs.push_back( &cObj );

	GLViewer::go( objs );
}

void compute_rings_redection(void){
	Image3D<short> im_short;
	bool falg = im_short.load( "data/vessel3d.data" );
	if(!falg) return;
	
	RD::mm_filter( im_short, 19 );
	im_short.save( "data/vessel3d.rd.19.data" );
}

void compute_center_line( string dataname = "roi15" ){
	Data3D<short> im_short;
	bool falg = im_short.load( "data/"+dataname+".data" );
	if(!falg) return;

	// vesselness
	Data3D<Vesselness_All> vn_all; 
	vn_all.load( "data/"+dataname+".vn_all" );

	MST::Graph3D<Edge> tree; 
	MST::edge_tracing( vn_all, tree, 0.55f, 0.055f );

	Data3D<unsigned char> im_unchar;
	IP::normalize( im_short, short(255) );
	im_short.convertTo( im_unchar );
	GLViewerExt::Volumn vObj( 
		im_unchar.getMat().data, 
		im_unchar.SX(), im_unchar.SY(), im_unchar.SZ() );

	GLViewer::CenterLine<Edge> cObj( tree );

	vector<GLViewer::Object*> objs;
	objs.push_back( &vObj );
	objs.push_back( &cObj ); 
	GLViewer::go( objs );

	return;
}	

void xuefeng_cut(void){	
	Image3D<short> im_short;
	im_short.load( "data/vessel3d.rd.19.data" );
	
	Vec3i piece_size(585,525,100);
	Vec3i pos(0,0,0);
	for( pos[2]=0; pos[2]+piece_size[2] <= im_short.SZ(); pos[2]+=piece_size[2]*3/4 ) {
		for( pos[1]=0; pos[1]+piece_size[1] <= im_short.SY(); pos[1]+=piece_size[1]*3/4 ) {
			for( pos[0]=0; pos[0]+piece_size[0] <= im_short.SX(); pos[0]+=piece_size[0]*3/4 )
			{
				cout << "Saveing ROI from " << pos << " to " << pos+piece_size-Vec3i(1,1,1) << endl;
				static int i = 0;
				stringstream ss;
				ss << "data/parts/vessel3d.rd.19.part" << i++ << ".data";
				im_short.setROI( pos, pos+piece_size-Vec3i(1,1,1) );
				im_short.saveROI( ss.str() );
			}
		}
	}
}


int main(int argc, char* argv[])
{
	// compute_vesselness( "roi16.partial", 0.5f, 7.5f, 0.3f );

	// Compute Center Line
	Data3D<Vesselness_All> vn_all; 
	vn_all.load( "output/roi16.partial.sigma_to7.5.vn_all" );
	MST::Graph3D<Edge> tree1; 
	MST::edge_tracing( vn_all, tree1, 0.55f, 0.055f );
	GLViewer::CenterLine<Edge> cObj1( tree1 );

	// Compute Center Line2
	Graph< MST::Edge_Ext, MST::LineSegment > tree2;
	MinSpanTree::build_tree_xuefeng( "data/roi16.partial.linedata.txt", tree2, 150 );
	GLViewer::CenterLine<MST::Edge_Ext, MST::LineSegment> cObj2( tree2 );
	
	vector<GLViewer::Object*> objs;
	objs.push_back( &cObj1 );
	objs.push_back( &cObj2 );

	GLViewer::VideoSaver videoSaver( "output/temp.avi" );
	GLViewer::go( objs, &videoSaver );

	// compute_min_span_tree_vesselness();
	// compute_center_line();

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

	////////////////////////////////////////////////////////////////
	// Plotting About Eigenvalues (Plots being used in my thesis)
	////////////////////////////////////////////////////////////////
	//// Visualization of the Eigenvalues
	// Validation::Eigenvalues::plot_1d_box();
	// Validation::Eigenvalues::plot_2d_tubes();
	// Validation::Eigenvalues::plot_2d_ball();
	// Validation::Eigenvalues::plot_3d_tubes();
	//// Draw Second Derivative of Gaussian on top of the Box function 
	// Validation::box_func_and_2nd_gaussian::plot_different_size();
	// Validation::box_func_and_2nd_gaussian::plot_different_pos();

	return 0;
}
