////////////////////////////////////////////////
// An Example of Visulizing the Data With Maximum Intensity Projection
#define _CRT_SECURE_NO_DEPRECATE

#include "stdafx.h"
#include "GLViwerWrapper.h"

GLViewerExt viwer;

int main(int argc, char* argv[])
{
	// Original Data (Before Rings Reduction) 
	Data3D<short> im_short; 
	// Load The Data
	bool flag = im_short.load( "data/roi16.partial.original.data" ); 
	if( !flag ) { 
		cout << "Load Data Fail." << endl; 
		return 0; 
	} 
	// Prepare the Data for Viewer 
	viwer.addObject( im_short, GLViewer::Volumn::MIP ); 
	// Go visualize the data with OpenGL 
	viwer.go(); 
	
	return 0; 
}
