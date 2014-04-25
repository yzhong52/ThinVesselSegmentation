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
		ver.go( 1280, 720, 2 );
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
	void initViwer( const Image3D<short>& im_short, const vector<cv::Vec3i>& dataPoints, 
		const vector<Line3D*>& lines, const vector<int>& labelings )
	{

	}
#endif



#include "SparseMatrix\SparseMatrix.h"
#include "ModelSet.h"

int main(int argc, char* argv[])
{
	srand( 3 ); 

	// TODO: not compatible with MinGW? 
	CreateDirectory(L"./output", NULL);
	
	// Vesselness measure with sigma
	Image3D<Vesselness_Sig> vn_sig;
	Image3D<short> im_short;
	vn_sig.load( "data/roi15.sigma_to8.vn_sig" ); 
	/*im_short.load( "../data/roi15.data" ); 
	return 0; */
	vn_sig.remove_margin_to( Vec3i(50, 50, 50) );
	
	// Synthesic Data
	//SyntheticData::Doughout( im_short ); 
	//SyntheticData::Stick( im_short ); 

	// threshold the data and put the data points into a vector
	Data3D<int> indeces;
	vector<cv::Vec3i> dataPoints;
	Data3D<float> vn = vn_sig; 
	IP::normalize( vn, 1.0f ); 
	
	IP::threshold( vn, indeces, dataPoints, 0.20f );
	
	//////////////////////////////////////////////////
	// Line Fitting
	//////////////////////////////////////////////////
	// Initial Samplings
	const int num_init_labels = (int) dataPoints.size(); 
	ModelSet<Line3D> model; 
	vector<Line3D*>& lines = model.models; 
	for( int i=0; i<num_init_labels; i++ ) {
		const Vec3i& dir = vn_sig.at( dataPoints[i] ).dir;
		const double& sigma = vn_sig.at( dataPoints[i] ).sigma;

		Line3DTwoPoint *line  = new Line3DTwoPoint();
		line->setPositions( dataPoints[i] - dir, dataPoints[i] + dir ); 
		line->setSigma( sigma ); 
		lines.push_back( line ); 
	}
	// model.deserialize<Line3DTwoPoint>( "output/Line3DTwoPoint.model" ); 
	if( lines.size()!=dataPoints.size() ) {
		cout << "Number of models is not corret. " << endl; 
		cout << "Probably because of errors while deserializing the data. " << endl;
		return 0; 
	}

	vector<int> labelings = vector<int>( dataPoints.size(), 0 ); 
	// randomly assign label for each point separatedly 
	for( int i=0; i<num_init_labels; i++ ) labelings[i] = i; 
	
	
	//////////////////////////////////////////////////
	// create a thread for rendering
	//////////////////////////////////////////////////
	initViwer( vn_sig, dataPoints, lines, labelings);
	
	LevenburgMaquart lm;
	
	Timer::begin( "Levenburg Maquart" ); 
	lm.reestimate( dataPoints, labelings, model, indeces ); 
	Timer::end( "Levenburg Maquart" ); 

	cout << "Main Thread is Done. " << endl; 
	cout << Timer::summery() << endl; 

	model.serialize( "output/Line3DTwoPoint.model" ); 

	WaitForSingleObject( thread_render, INFINITE);
	return 0; 
}