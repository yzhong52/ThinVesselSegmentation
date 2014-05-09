#pragma once

/////////////////////////////////////
// Glew Library 
#include "gl\glew.h"  // For Texture 3D and Blending_Ext
#pragma comment(lib, "glew32.lib")

/////////////////////////////////////
// OpenGL Library
#include <windows.h>		// Header File For Windows
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library




#include "GLVideoSaver.h"
#include <vector>

class GLCamera; 

namespace GLViewer
{
	// derive the following virtual class in order to render your own objects
	class Object{
	public:
		// Pure virtual functions 
		// you have to implement these virtual functions in order to render it 
		// with GLViewer
		virtual void render(void) = 0;				 // render the object
		virtual unsigned int size_x(void) const = 0; // size of the object
		virtual unsigned int size_y(void) const = 0; // size of the object
		virtual unsigned int size_z(void) const = 0; // size of the object

		// Optional funtions to overide
		// init function for OpenGL, excuted before rendering loop 
		virtual void init(void) { }
		// render function for OpenGL
		// keyboard function for OpenGL
		virtual void keyboard( unsigned char key ) { }
	}; 

	// camera infos
	extern GLCamera cam;
	extern int numViewports; 
	void go( std::vector<Object*> objects, int w=1280, int h = 720  ); 

	void startCaptureVideo( int maxNumFrames = 3600 ); 

	// TODO: visualize color texture
}

