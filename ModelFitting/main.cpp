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
#include "GLViwerModel.h"
#include "MinSpanTree.h" 

// For the use of graph cut
#include "GCoptimization.h"
typedef GCoptimization GC; 

#include "Line3DTwoPoint.h" 
#include "LevenburgMaquart.h" 
#include "GraphCut.h"

// for visualization
GLViwerModel ver;

void visualization_func( void* data ) {
	GLViwerModel& ver = *(GLViwerModel*) data; 
	ver.go();
}

const double LOGLIKELIHOOD = 100; 

int main(int argc, char* argv[])
{
	CreateDirectory(L"./output", NULL);
	
	Data3D<short> im_short;
	//Synthesic Data
	im_short.reset( Vec3i(20,20,20) ); 
	/*for( int i=5; i<15; i++ ) {
		im_short.at(i,  i,  i)   = 100; 
		im_short.at(i,  i,  i+1) = 100; 
		im_short.at(i,  i+1,i)   = 100; 
		im_short.at(i+1,i,  i)   = 100; 
	}*/
	im_short.at(5, 5, 5) = 10000; 
	im_short.at(5, 5, 15) = 10000; 
	im_short.at(5, 6, 15) = 10000; 
	im_short.at(15, 16, 15) = 10000; 

	// OR real data
	//im_short.load( "../data/data15.data" );

	// threshold the data and put the data points into a vector
	Data3D<unsigned char> im_uchar;
	vector<cv::Vec3i> dataPoints;
	IP::threshold( im_short, im_uchar, dataPoints, short(50) );

	GLViewer::GLLineModel *model = new GLViewer::GLLineModel( im_short.get_size() );
	ver.objs.push_back( model );

	//////////////////////////////////////////////////
	// create a thread for rendering
	//////////////////////////////////////////////////
	HANDLE thread_render = NULL; 
	thread_render = (HANDLE) _beginthread( visualization_func, 0, (void*)&ver ); 
	
	model->updatePoints( dataPoints ); 

	//////////////////////////////////////////////////
	// Line Fitting
	//////////////////////////////////////////////////
	// Initial Samplings
	const int num_init_labels = 1; 
	vector<Line3D*> lines; 
	Line3DTwoPoint* line = new Line3DTwoPoint();
	line->setPositions( Vec3i(5,6,5), Vec3i(5,4,10) ); 
	lines.push_back( line );  
	Line3DTwoPoint* line2 = new Line3DTwoPoint();
	line2->setPositions( Vec3i(5,6,13), Vec3i(15,16,12) ); 
	lines.push_back( line2 );  

	vector<int> labelings = vector<int>( dataPoints.size(), 0 ); 
	labelings[2] = 1; 
	labelings[3] = 1; 

	model->updateModel( lines, labelings ); 

	cout << "Graph Cut Begin" << endl; 
	try{
		// keep track of energy in previous iteration
		GC::EnergyType energy_before = -1;

		// TODO: let's run the algorithm for only one iteration for now
		for( int i=0; i<1; i++ ) { 
			// TODO: let's not have background model for now. We will add background model later
			
			// GC::EnergyType energy = GraphCut::estimation( dataPoints, labelings, lines ); 
			
			model->updateModel( lines, labelings ); 
			LevenburgMaquart::reestimate( dataPoints, labelings, lines ); 
		}
	}
	catch (GCException e){
		e.Report();
	}
	
	cout << "Main Thread is Done. " << endl; 
	WaitForSingleObject( thread_render, INFINITE);

	for( int i=0; i<num_init_labels; i++ ){
		delete lines[i]; 
		lines[i] = NULL; 
	}

	return 0; 
}