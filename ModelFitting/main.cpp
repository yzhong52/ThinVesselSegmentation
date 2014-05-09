// ModelFitting.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <Windows.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "Line3D.h"

// For multithreading
#include <Windows.h>
#include <process.h>
#include <iomanip>

// This project is build after VesselNess. 
// Some of the building blocks (Data3D, Visualization) are borrowed from VesselNess. 
#include "Data3D.h" 

#include "MinSpanTree.h" 

// For the use of graph cut
#include "GCoptimization.h"
typedef GCoptimization GC; 

#include "Line3DTwoPoint.h" 
#include "LevenburgMaquart.h" 
#include "GraphCut.h"
#include "SyntheticData.h" 
#include "ImageProcessing.h"
#include "Timer.h"
#include "Neighbour26.h"
#include "SparseMatrix\SparseMatrix.h"
#include "ModelSet.h"
#include "init_models.h" 

#include <assert.h>
#include <iostream>
#include <limits> 

const double LOGLIKELIHOOD = 1.0; 
const double PAIRWISESMOOTH = 7.0; 


// if IS_PROFILING is defined, rendering is disabled
//#define IS_PROFILING
HANDLE thread_render = NULL; 

#ifndef IS_PROFILING // NOT profiling, add visualization model
#include "GLViwerModel.h"
GLViwerModel ver;
// thread function for rendering
void visualization_func( void* data ) {
	GLViwerModel& ver = *(GLViwerModel*) data; 
	ver.go( 720, 480, 2 );
}
template<class T>
void initViwer( const Data3D<T>& im, const vector<cv::Vec3i>& dataPoints, 
	const vector<Line3D*>& lines, const vector<int>& labelings )
{
	GLViewer::GLLineModel *model = new GLViewer::GLLineModel( im.get_size() );
	model->updatePoints( dataPoints ); 
	model->updateModel( lines, labelings ); 
	ver.objs.push_back( model );

	ver.addObject( im ); 

	thread_render = (HANDLE) _beginthread( visualization_func, 0, (void*)&ver ); 
	// Give myself sometime to decide whether we need to render a video
	Sleep( 1000 ); 
}
#else
template<class T>
void initViwer( const Image3D<T>& im_short, const vector<cv::Vec3i>& dataPoints, 
	const vector<Line3D*>& lines, const vector<int>& labelings )
{

}
#endif

//
//
//
//void each_model_per_point( 
//	const Image3D<Vesselness_Sig>& vn_sig,
//	Data3D<int>& labelID3d, 
//	vector<cv::Vec3i>& tildaP,
//	ModelSet<Line3D>& model, 
//	vector<int>& labelID )
//{
//	Data3D<float> vn = vn_sig; 
//	IP::normalize( vn, 1.0f ); 
//
//	tildaP.clear(); 
//	labelID.clear();
//	labelID3d.reset( vn.get_size(), -1 );
//	model.models.clear(); 
//	
//	for(int z=0;z<vn.SZ();z++) for ( int y=0;y<vn.SY();y++) for(int x=0;x<vn.SX();x++) {
//		if( vn.at(x,y,z) > 0.1f ) { // a thread hold
//			int lid = (int) model.models.size(); 
//
//			labelID3d.at(x,y,z) = lid; 
//
//			labelID.push_back( lid ); 
//
//			tildaP.push_back( Vec3i(x,y,z) ); 
//
//			const Vec3d pos(x,y,z); 
//			const Vec3d& dir = vn_sig.at(x,y,z).dir;
//			const double& sigma = vn_sig.at(x,y,z).sigma;
//			Line3DTwoPoint *line  = new Line3DTwoPoint();
//			line->setPositions( pos-dir, pos+dir ); 
//			line->setSigma( sigma ); 
//			model.models.push_back( line ); 
//		}
//	}
//}
//
//
//void each_model_per_local_maximum( 
//	const Data3D<Vesselness_Sig>& vn_sig,
//	Data3D<int>& labelID3d, 
//	vector<cv::Vec3i>& tildaP,
//	ModelSet<Line3D>& model, 
//	vector<int>& labelID )
//{
//	Data3D<float> vn = vn_sig; 
//	IP::normalize( vn, 1.0f ); 
//
//	Data3D<Vec3i> djs( vn.get_size() );
//
//	for(int z=0;z<vn.SZ();z++) for ( int y=0;y<vn.SY();y++) for(int x=0;x<vn.SX();x++) {
//		// find the major orientation
//		// assigning the orientation to one of the 13 categories
//		const Vec3f& cur_dir = vn_sig.at(x,y,z).dir;
//		int mdi = 0; // major direction id
//		float max_dot_product = 0;
//		for( int di=0; di<13; di++ ){
//			float current_dot_product = cur_dir.dot( Neighbour26::at(di) );
//			if( max_dot_product < current_dot_product ) { 
//				max_dot_product = current_dot_product;
//				mdi = di;// update the major direction id
//			}
//		}
//		const vector<cv::Vec3i>& cross_section = Neighbour26::getCrossSection(mdi); 
//		Vec3i mics( -1, -1, -1 ); // maximum index on cross section 
//		float mrsp = vn.at(x,y,z); // maximum response
//		for( int i =0; i < cross_section.size(); i++ ){
//			Vec3i offset_pos  = Vec3i(x,y,z) + cross_section[i];
//			if( vn.isValid( offset_pos ) && mrsp < vn.at( offset_pos ) ) {
//				mrsp = vn.at(offset_pos); 
//				mics = offset_pos; 
//			} 
//		}
//		// 'mics==Vec3i( -1, -1, -1 )' indicates that the current position is local maximum 
//		djs.at(x,y,z) = mics;
//	}
//
//
//	tildaP.clear(); 
//	labelID.clear();
//	labelID3d.reset( vn.get_size(), -1 );
//	model.models.clear(); 
//	for(int z=0;z<vn.SZ();z++) for ( int y=0;y<vn.SY();y++) for(int x=0;x<vn.SX();x++) {
//		if( vn.at(x,y,z) > 0.1f && djs.at(x,y,z)==Vec3i( -1, -1, -1 ) ) { 
//			int lid = (int) model.models.size(); 
//
//			labelID3d.at(x,y,z) = lid; 
//
//			labelID.push_back( lid ); 
//
//			tildaP.push_back( Vec3i(x,y,z) ); 
//
//			const Vec3d pos(x,y,z); 
//			const Vec3d& dir = vn_sig.at(x,y,z).dir;
//			const double& sigma = vn_sig.at(x,y,z).sigma;
//			Line3DTwoPoint *line  = new Line3DTwoPoint();
//			line->setPositions( pos-dir, pos+dir ); 
//			line->setSigma( sigma ); 
//			model.models.push_back( line ); 
//		}
//	}
//
//	for(int z=0;z<vn.SZ();z++) for ( int y=0;y<vn.SY();y++) for(int x=0;x<vn.SX();x++) {
//		Vec3i pos(x,y,z); 
//		while( djs.at( pos )!=Vec3i( -1, -1, -1 ) ) { 
//			pos = djs.at( pos ); 
//		}
//		if( labelID3d.at(pos)!=-1 ) {
//			labelID3d.at(x,y,z) = labelID3d.at(pos);
//			tildaP.push_back( Vec3i(x,y,z) ); 
//			labelID.push_back( labelID3d.at(pos) ); 
//		}
//	}
//}

int main(int argc, char* argv[])
{
	srand( 3 ); 

	// TODO: not compatible with MinGW? 
	CreateDirectory(L"./output", NULL);

	// Vesselness measure with sigma
	Image3D<Vesselness_Sig> vn_sig;
	// Image3D<Vesselness_Sig> vn_sig_nms;
	
	vn_sig.load( "../temp/data15.vn_sig" ); 
	vn_sig.remove_margin_to( Vec3i(50, 50, 50) );

	// threshold the data and put the data points into a vector
	Data3D<int> labelID3d; 
	vector<cv::Vec3i> tildaP; 
	ModelSet<Line3D> model; 
	vector<int> labelID; 
	each_model_per_point( vn_sig, labelID3d, tildaP, model, labelID ); 
	// each_model_per_local_maximum( vn_sig, labelID3d, tildaP, model, labelID ); 

	//////////////////////////////////////////////////
	// create a thread for rendering
	//////////////////////////////////////////////////
	initViwer( vn_sig, tildaP, model.models, labelID );

	//////////////////////////////////////////////////
	// Levenberg-Marquart
	//////////////////////////////////////////////////
	Timer::begin( "Levenberg Marquart" ); 
	cout << "Number of data points: " << tildaP.size() << endl;
	LevenburgMaquart lm( tildaP, labelID, model, labelID3d );
	lm.reestimate( 4000 ); 
	Timer::end( "Levenberg Marquart" ); 

	cout << Timer::summery() << endl; 

	model.serialize( "output/Line3DTwoPoint.model" ); 

	cout << "Main Thread is Done. " << endl; 
	WaitForSingleObject( thread_render, INFINITE);
	return 0; 
}


//////////////////////////////////////////////////
// Loading serialized data
//////////////////////////////////////////////////
//model.deserialize<Line3DTwoPoint>( "output/Line3DTwoPoint.model" ); 
//if( lines.size()!=dataPoints.size() ) {
//	cout << "Number of models is not corret. " << endl; 
//	cout << "Probably because of errors while deserializing the data. " << endl;
//	return 0; 
//}
