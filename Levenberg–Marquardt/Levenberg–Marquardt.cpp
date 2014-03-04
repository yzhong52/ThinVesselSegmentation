// Levenberg¨CMarquardt.cpp : Defines the entry point for the console application.
//

// This project is build after VesselNess. 
// Some of the building blocks (Data3D, Visualization) are borrowed from VesselNess. 

#include "Data3D.h"
#include "GLViwerWrapper.h" // For visualization

#include <iostream>
using namespace std;

GLViewerExt ver;

int main(int argc, char* argv[])
{
	Data3D<short> im_short;


	im_short.load( "../data/data15.data" );
	ver.addObject( im_short ); 
	ver.go();

	return 0;
}

