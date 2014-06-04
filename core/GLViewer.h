#pragma once

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




#include "GLVideoSaver.h"
#include <vector>

class GLCamera;

namespace GLViewer
{
// derive the following virtual class in order to render your own objects
class Object
{
public:
    Object() {}
    virtual ~Object() {}
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

