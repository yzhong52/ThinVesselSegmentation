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

const double LOGLIKELIHOOD = 1.15; 
const double PAIRWISESMOOTH = 3.0; 



#define IS_PROFILING
HANDLE thread_render = NULL; 
#ifndef IS_PROFILING // NOT profiling, add visualization model
	#include "GLViwerModel.h"
	GLViwerModel ver;
	// thread function for rendering
	void visualization_func( void* data ) {
		GLViwerModel& ver = *(GLViwerModel*) data; 
		ver.go();
	}
	void initViwer( const Image3D<short>& im_short, const vector<cv::Vec3i>& dataPoints, 
		const vector<Line3D*>& lines, const vector<int>& labelings )
	{
		GLViewer::GLLineModel *model = new GLViewer::GLLineModel( im_short.get_size() );
		model->updatePoints( dataPoints ); 
		model->updateModel( lines, labelings ); 
		ver.objs.push_back( model );
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
int main(int argc, char* argv[])
{

	srand( 3 ); 

	// TODO: not compatible with MinGW? 
	// CreateDirectory(L"./output", NULL);
	
	Image3D<short> im_short;
	//// Real data
	//im_short.load( "../data/data15.data" );
	//im_short.shrink_by_half();

	// Synthesic Data
	//SyntheticData::Doughout( im_short ); 
	SyntheticData::Stick( im_short ); 

	// threshold the data and put the data points into a vector
	Data3D<int> indeces;
	vector<cv::Vec3i> dataPoints;
	IP::threshold( im_short, indeces, dataPoints, short(4500) );
	
	cout << "Number of data points: "  << dataPoints.size() << endl; 
	
	//////////////////////////////////////////////////
	// Line Fitting
	//////////////////////////////////////////////////
	// Initial Samplings
	const int num_init_labels = (int) dataPoints.size(); 
	vector<Line3D*> lines; 
	for( int i=0; i<num_init_labels; i++ ) {
		Line3DTwoPoint *line  = new ::Line3DTwoPoint();
		Vec3i randomDir = Vec3i(
				rand() % 200 - 100, 
				//rand() % 200 - 100, 
				// rand() % 100 + 10, 
				rand() % 100 + 10, 
				rand() % 100 + 10 ); 
		line->setPositions( dataPoints[i] - randomDir + Vec3i(2, 3, 1) , dataPoints[i] + randomDir + Vec3i(3, 2, 1) ); 
		lines.push_back( line ); 
	}
	
	vector<int> labelings = vector<int>( dataPoints.size(), 0 ); 
	// randomly assign label for each point separatedly 
	for( int i=0; i<num_init_labels; i++ ) labelings[i] = i; 
	
	
	//////////////////////////////////////////////////
	// create a thread for rendering
	//////////////////////////////////////////////////
	initViwer( im_short, dataPoints, lines, labelings);
	
	LevenburgMaquart lm;
	
	Timmer::begin();
	lm.reestimate( dataPoints, labelings, lines, indeces ); 
	Timmer::end(); 

	cout << "Main Thread is Done. " << endl; 
	cout << Timmer::summery() << endl; 

	WaitForSingleObject( thread_render, INFINITE);

	for( int i=0; i<num_init_labels; i++ ){
		delete lines[i]; 
		lines[i] = NULL; 
	}

	return 0; 
}