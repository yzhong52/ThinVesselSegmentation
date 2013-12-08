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

/////////////////////////////////////
// Glut Library
#include "GL\freeglut.h"
#pragma comment(lib, "freeglut.lib")


#include "VideoSaver.h"
#include <vector>

class GLCamera; 

namespace GLViewer
{
	class Object{
	public:
		// init function for OpenGL
		virtual void init(void) { };
		// render function for OpenGL
		virtual void render(void) = 0;
		// keyboard function for OpenGL
		virtual void keyboard(unsigned char key ) { }
		// size of the object
		virtual unsigned int size_x(void) const = 0;
		virtual unsigned int size_y(void) const = 0;
		virtual unsigned int size_z(void) const = 0;
	}; 

	// camera infos
	extern GLCamera cam;
	/*extern GLfloat vec_y[3];
	extern GLfloat vec_x[3];
	extern GLfloat t[3];*/
	void go( std::vector<Object*> objects, VideoSaver* videoSaver = NULL ); 

	// TODO: to visualized color texture
}

