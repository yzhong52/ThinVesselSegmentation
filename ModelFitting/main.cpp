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
#include <opencv2\core\core.hpp>

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

#if !(_MSC_VER && !__INTEL_COMPILER)
	#define IS_PROFILING
#endif

#ifndef IS_PROFILING // NOT profiling, add visualization model
	#include "GLViwerModel.h"
	GLViwerModel ver;
	// thread function for rendering
	void visualization_func( void* data ) {
		const int& numViewports = *(int*) data;
		ver.go( 1280, 768, numViewports );
	}

	// unfortunately! fucntiona templates is not supported only in C++11
	template<class T1>
	void initViwer( const vector<cv::Vec3i>& dataPoints,
		const vector<Line3D*>& lines, const vector<int>& labelings,
		const Data3D<T1>* im1 )
	{
		initViwer( dataPoints, lines, labelings, im1,
			(Data3D<char>*) NULL,
			(Data3D<char>*) NULL );
	}

	template<class T1, class T2, class T3>
	void initViwer( const vector<cv::Vec3i>& dataPoints,
		const vector<Line3D*>& lines, const vector<int>& labelings,
		const Data3D<T1>* im1,
		const Data3D<T2>* im2,
		const Data3D<T3>* im3 )
	{
		ver.addObject( *im1 );

		if( im2 ) ver.addObject( *im2 );
		if( im3 ) ver.addObject( *im3 );

		GLViewer::GLLineModel *model = new GLViewer::GLLineModel( im1->get_size() );
		model->updatePoints( dataPoints );
		model->updateModel( lines, labelings );
		ver.objs.push_back( model );

		int numViewports = 2;
		numViewports += (im2!=NULL);
		numViewports += (im3!=NULL);
		thread_render = (HANDLE) _beginthread( visualization_func, 0, (void*)&numViewports );

		Sleep( 1000 );
	}

#else
    template<class T1>
	void initViwer( const vector<cv::Vec3i>& dataPoints,
		const vector<Line3D*>& lines, const vector<int>& labelings,
		const Data3D<T1>* im1 ) { }

	template<class T1, class T2, class T3>
	void initViwer( const vector<cv::Vec3i>& dataPoints,
		const vector<Line3D*>& lines, const vector<int>& labelings,
		const Data3D<T1>* im1,
		const Data3D<T2>* im2,
		const Data3D<T3>* im3 ) { }
#endif


void experiment6_video( void ) {
	// Vesselness measure with sigma
	Image3D<Vesselness_Sig> vn_et_sig;
	vn_et_sig.load( "../temp/data15.et.vn_sig" );

	// threshold the data and put the data points into a vector
	Data3D<int> labelID3d;
	vector<cv::Vec3i> tildaP;
	ModelSet<Line3D> model;
	vector<int> labelID;
	each_model_per_point( vn_et_sig, labelID3d, tildaP, model, labelID );

	Data3D<Vesselness_Sig> vn_sig( "../temp/data15.vn_sig" );
	Data3D<short> im_short( "../data/data15.data" );

	// create a thread for rendering
	initViwer( tildaP, model.models, labelID, &im_short, &vn_sig, &vn_et_sig );

	cout << "Number of data points: " << tildaP.size() << endl;

	// Levenberg-Marquart
	LevenburgMaquart lm( tildaP, labelID, model, labelID3d );
	lm.reestimate( 4000, LevenburgMaquart::Quadratic );

	cout << "Main Thread is Done. " << endl;
	WaitForSingleObject( thread_render, INFINITE);
}


void experiment1_video( void ) {
	// Vesselness measure with sigma
	Data3D<Vesselness_Sig> vn_sig( "../temp/yes.vn_sig" );

	// threshold the data and put the data points into a vector
	Data3D<int> labelID3d;
	vector<cv::Vec3i> tildaP;
	ModelSet<Line3D> model;
	vector<int> labelID;
	each_model_per_point( vn_sig, labelID3d, tildaP, model, labelID );

	// create a thread for rendering
	initViwer( tildaP, model.models, labelID, &vn_sig );

	// Levenberg-Marquart
	cout << "Number of data points: " << tildaP.size() << endl;

	LevenburgMaquart lm( tildaP, labelID, model, labelID3d );
	lm.reestimate( 1000, LevenburgMaquart::Quadratic );
	system( "pause" );
	lm.reestimate( 4000, LevenburgMaquart::Linear );

	cout << "Main Thread is Done. " << endl;
	WaitForSingleObject( thread_render, INFINITE);
}



void test1_twopoints( void ) {
	// Vesselness measure with sigma
	Data3D<char> im_uchar( Vec3i(10, 10, 10), 0 );
	im_uchar.at(5,5,5) = 20; 
	im_uchar.at(5,6,6) = 20; 
	im_uchar.at(6,7,7) = 20; 

	// threshold the data and put the data points into a vector
	Data3D<int> labelID3d( im_uchar.get_size(), -1 );

	vector<cv::Vec3i> tildaP;
	ModelSet<Line3D> model;
	vector<int> labelID;

	for(int z=0;z<im_uchar.SZ();z++) for ( int y=0;y<im_uchar.SY();y++) for(int x=0;x<im_uchar.SX();x++) {
		if( im_uchar.at(x,y,z) > 10 ) { // a thread hold
			int lid = (int) model.models.size();

			labelID3d.at(x,y,z) = lid;

			labelID.push_back( lid );

			tildaP.push_back( Vec3i(x,y,z) );

			const Vec3d pos(x,y,z);
			const Vec3d dir( rand()%100, rand()%100, rand()%100 + 1); 
			dir / sqrt( dir.dot( dir ) );
			const double sigma = 1;
			Line3DTwoPoint *line  = new Line3DTwoPoint();
			line->setPositions( pos-dir, pos+dir );
			line->setSigma( sigma );
			model.models.push_back( line );
		}
	}

	// Levenberg-Marquart
	cout << "Number of data points: " << tildaP.size() << endl;

	LevenburgMaquart lm( tildaP, labelID, model, labelID3d );
	lm.reestimate( 1000, LevenburgMaquart::Quadratic );
	system( "pause" );
	lm.reestimate( 4000, LevenburgMaquart::Linear );

	cout << "Main Thread is Done. " << endl;
	WaitForSingleObject( thread_render, INFINITE);
}

int main(int argc, char* argv[])
{
    cout << Mat::zeros(3,2, CV_32F) << endl;
	experiment1_video();
	return 0;
}


//// TODO: not compatible with MinGW?
//CreateDirectory(L"./output", NULL);

//////////////////////////////////////////////////
// Loading serialized data
//////////////////////////////////////////////////
//model.deserialize<Line3DTwoPoint>( "output/Line3DTwoPoint.model" );
//if( lines.size()!=dataPoints.size() ) {
//	cout << "Number of models is not corret. " << endl;
//	cout << "Probably because of errors while deserializing the data. " << endl;
//	return 0;
//}
