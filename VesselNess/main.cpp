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
#include "Validation.h" 

GLViewerExt viwer;

void compute_and_save_vesselness( string dataname = "vessel3d.rd.19",
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
	vesselness_log << "Vesselness is computed with sigmas: ";
	for( float sigma = sigma_from; sigma < sigma_to; sigma += sigma_step ) {
		vesselness_log << sigma << ","; 
	}

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
}

void compute_min_span_tree( string data_name = "roi16.partial" ) {
	//Data3D<short> im_short;

	//im_short.load( "data/" +data_name+ ".data" );
	//
	//Image3D<unsigned char> im_uchar;
	//IP::normalize( im_short, short(255) );
	//im_short.convertTo( im_uchar );
	//
	//vector<GLViewer::Object*> objs;
	//GLViewer::Volumn vObj( im_uchar.getMat().data, 
	//	im_uchar.SX(), im_uchar.SY(), im_uchar.SZ() );
	//objs.push_back( &vObj );

	//// Computer Min Span Tree
	//Graph< MST::Edge_Ext, MST::LineSegment > tree;
	//MinSpanTree::build_tree_xuefeng( "data/" +data_name+ ".linedata.txt", tree, 150 );

	//GLViewer::CenterLine<MST::Edge_Ext, MST::LineSegment> cObj( tree );
	//objs.push_back( &cObj );

	//GLViewer::VideoSaver videoSaver( "output/line fitting.avi" );
	//GLViewer::go( objs/*, &videoSaver*/ );
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
	GLViewer::Volumn vObj( 
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
	Image3D<Vesselness_Sig> vn_sig; 
	vn_sig.load( "data/vessel3d.rd.19.sigma45.vn_sig" );

	Vec3i piece_size(585,525,100);
	Vec3i pos(0,0,0);
	for( pos[2]=0; pos[2]+piece_size[2] <= im_short.SZ(); pos[2]+=piece_size[2]*3/4 ) {
		for( pos[1]=0; pos[1]+piece_size[1] <= im_short.SY(); pos[1]+=piece_size[1]*3/4 ) {
			for( pos[0]=0; pos[0]+piece_size[0] <= im_short.SX(); pos[0]+=piece_size[0]*3/4 )
			{
				cout << "Saveing ROI from " << pos << " to " << pos+piece_size-Vec3i(1,1,1) << endl;
				static int i = 0;
				stringstream ss;
				ss << "data/parts/vessel3d.rd.19.part" << i++;
				im_short.setROI( pos, pos+piece_size-Vec3i(1,1,1) );
				vn_sig.setROI(  pos, pos+piece_size-Vec3i(1,1,1) );
				im_short.saveROI( ss.str()  + ".data");
				vn_sig.saveROI(  ss.str() + ".vn_sig");
			}
		}
	}
}


void load_graph( MST::Graph3D<Edge>& graph, const string& filename ) {
	ifstream fin;
	fin.open( filename );

	int sx, sy, sz;
	fin >> sx;
	fin >> sy;
	fin >> sz;

	graph.sx = sx;
	graph.sy = sy;
	graph.sz = sz;
	graph.reset( sx*sy*sz );

	unsigned int num_edges;
	fin >> num_edges; 
	for( unsigned int i=0; i<num_edges; i++ ) {
		Edge e;
		fin >> e.node1;
		fin >> e.node2;
		fin >> e.weight;
		graph.add_edge( e );
	}
}


void save_graph( MST::Graph3D<Edge>& graph, const string& filename ) {
	ofstream fout;
	fout.open( filename );
	fout << graph.sx << " ";
	fout << graph.sy << " ";
	fout << graph.sz << endl;
	fout << graph.num_edges() << endl;
	for( unsigned int i=0; i<graph.num_edges(); i++ ) {
		fout << graph.get_edge(i).node1 << " ";
		fout << graph.get_edge(i).node2 << " ";
		fout << graph.get_edge(i).weight << endl;
	}
}

int main(int argc, char* argv[])
{
	//Validation::box_func_and_2nd_gaussian::plot_different_size();
	//Validation::box_func_and_2nd_gaussian::plot_different_pos();
	
	//Validation::Eigenvalues::plot_1d_box();
	Validation::Eigenvalues::plot_2d_tubes();
	// Validation::Eigenvalues::plot_2d_ball();
	//Validation::Eigenvalues::plot_3d_tubes();
	return 0;

	// Vesselness for different sigmas
	 Data3D<float> vn_float1, vn_float2, vn_float3, vn_float4;
	vn_float1.load( "output/roi16.partial.sigma_to0.8.vn_float" );  viwer.addObject( vn_float1 ); 
	vn_float2.load( "output/roi16.partial.sigma_to1.3.vn_float" );  viwer.addObject( vn_float2 ); 
	vn_float3.load( "output/roi16.partial.sigma_to2.6.vn_float" );  viwer.addObject( vn_float3 ); 
	vn_float4.load( "output/roi16.partial.sigma_to5.1.vn_float" );  viwer.addObject( vn_float4 ); 

	// Original Data (Before Rings Reduction) 
	Data3D<short> im_short0;
	im_short0.load( "data/roi16.partial.original.data" );
	viwer.addObject( im_short0 );

	//// Original Data (After Rings Reduction) 
	Image3D<short> im_short;
	im_short.load( "data/roi16.partial.data" );
	viwer.addObject( im_short );

	//// Vesselness
	//Data3D<Vesselness_All> vn_all;
	////vn_all.load( "data/roi16.rd.19.sigma45.vn_all" );
	//vn_all.load( "data/roi16.partial.sigma_to8.vn_all" );
	//viwer.addObject( vn_all );
	
	// Direction of Vesselness
	Data3D<Vesselness_Sig> vn_sig;
	vn_sig.load( "data/roi16.partial.sigma_to8.vn_sig" );
	viwer.addObject( vn_sig );


	// Direction of Vesselness
	Data3D<float> vn_float;
	vn_float.load( "data/roi16.partial.sigma_to8.vn_float" );
	viwer.addObject( vn_float );


	// Ring reduction after model fitting
	//Graph< MST::Edge_Ext, MST::LineSegment > rings;
	// pre_process_xuefeng( "data/roi16.partial", "data/roi16.partial.rd", rings, 
	//	/*Center of Rings*/ MST::Vec3f(234-120, 270-89, 0) );
	//GLViewer::CenterLine<MST::Edge_Ext, MST::LineSegment> *cObj = viwer.addObject( rings );
	//cObj->setColor( 1.0f, 1.0f, 0.0f,/*Yellow*/ 1.0f, 1.0f, 0.0f/*Yellow*/ );
	
	//Graph< MST::Edge_Ext, MST::LineSegment > tree;
	//MinSpanTree::build_tree_xuefeng( "data/roi16.partial.rd", tree, 250 );
	//viwer.addObject( tree );
	
	//MST::Graph3D<Edge> tree2; 
	//// MST::edge_tracing( vn_all, tree2, 0.55f, 0.015f ); 
	//// save_graph( tree2, "output/roi16.rd.19.sigma45.edge_tracing.min_span_tree.txt" );
	//load_graph( tree2, "output/roi16.rd.19.sigma45.edge_tracing.min_span_tree.txt" );
	//viwer.addObject( tree2 ); 
	
	viwer.go();

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
