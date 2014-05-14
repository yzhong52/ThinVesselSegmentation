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
#include "EnergyFunctions.h" 

#include <assert.h>
#include <iostream>
#include <limits> 

const double DATA_COST = 1.0; 
const double PAIRWISE_SMOOTH = 7.0; 
const double DATA_COST2 = DATA_COST * DATA_COST; 
const double PAIRWISE_SMOOTH2 = PAIRWISE_SMOOTH * PAIRWISE_SMOOTH; 

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
}
#else
template<class T>
void initViwer( const Image3D<T>& im_short, const vector<cv::Vec3i>& dataPoints, 
	const vector<Line3D*>& lines, const vector<int>& labelings )
{

}
#endif

int main(int argc, char* argv[])
{
	srand( 3 ); 

	// TODO: not compatible with MinGW? 
	CreateDirectory(L"./output", NULL);

	// Vesselness measure with sigma
	Image3D<Vesselness_Sig> vn_sig;
	// Image3D<Vesselness_Sig> vn_sig_nms;
	
	// vn_sig.load( "../temp/data15.vn_sig" ); 
	vn_sig.load( "../temp/yes.vn_sig" ); 
	vn_sig.remove_margin_to( Vec3i(20, 10, 10) );

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
	system( "pause" ); 
	lm.reestimate_abs_esp( 40000 ); 

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
