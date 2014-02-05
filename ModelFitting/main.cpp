// ModelFitting.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// This project is build after VesselNess. 
// Some of the building blocks (Data3D, Visualization) are borrowed from VesselNess. 

#include "Data3D.h" 
#include "GLViwerWrapper.h"
#include <Windows.h>


GLViewerExt ver;

int main(int argc, char* argv[])
{
	CreateDirectory(L"./output", NULL);

	// TODO: Fix this function.
	Data3D<short> im_short;
	im_short.load( "data/roi15.data" );
	ver.addObject( im_short, GLViewer::Volumn::MIP ); 
	ver.go();

	return 0;
}

