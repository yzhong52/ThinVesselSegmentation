// Levenberg¨CMarquardt.cpp : Defines the entry point for the console application.
//

// This project is build after VesselNess. 
// Some of the building blocks (Data3D, Visualization) are borrowed from VesselNess. 

#include "Data3D.h"         // For data manipulation
#include "GLViwerWrapper.h" // For visualization
#include "ImageProcessing.h"// For image processing

#include <iostream>
using namespace std;

GLViewerExt ver;

struct Offset {
	float theta1;
	float theta2;
	float distance;
}; 

int main(int argc, char* argv[])
{
	Data3D<short> im_short;
	bool flag = im_short.load( "../data/data15.data" );
	if( !flag ) return 0;

	Data3D<unsigned char> im_mask;
	vector<Vec3i> poss;
	IP::threshold( im_short, im_mask, poss, short(4300) ); // TODO: should try 3300 later


	// random sampling for each of the data points 
	// each point will have three properties 
	

	
	ver.addObject( im_short, GLViewer::Volumn::MIP ); 
	ver.addObject( im_mask, GLViewer::Volumn::MIP ); 
	ver.go();

	return 0;
}

