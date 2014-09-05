#pragma once
#include <vector>

/////////////////////////////////////
// Glew Library
#include <GL/glew.h> // For Texture 3D and Blending_Ext

/////////////////////////////////////
// OpenGL Library
#if _MSC_VER && !__INTEL_COMPILER
#include <windows.h>		// Header File For Windows
#include <GL\GL.h>			// Header File For The OpenGL32 Library
#include <GL\glu.h>			// Header File For The GLu32 Library
#endif

#include "GLObject.h"

class GLCamera;

namespace GLViewer
{

extern GLCamera camera;

// number of viewports
extern int numViewports;

// Display a given list of 'objects' in a windows of size ['w', 'h']
void dispay( std::vector<Object*> objects, int w=1280, int h = 720  );

// API for start capture a video clip
void startCaptureVideo( int maxNumFrames = 3600 );

}

