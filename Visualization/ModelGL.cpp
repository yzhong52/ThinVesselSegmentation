///////////////////////////////////////////////////////////////////////////////
// ModelGL.cpp
// ===========
// Model component of OpenGL
// 
// AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2006-07-10
// UPDATED: 2013-11-27
///////////////////////////////////////////////////////////////////////////////

#if _WIN32
    #include <windows.h>    // include windows.h to avoid thousands of compile errors even though this class is not depending on Windows
	#include "Log.h"
#endif


#include "GLViewer.h" // Yuchen
#include "Volumn.h" // Yuchen

#include "glew.h" // Yuchen
#pragma comment(lib, "glew32.lib") //Yuchen

#include <GL/gl.h>
#include <GL/glu.h>
#include "ModelGL.h"
#include "Bmp.h"
#include "smart_assert.h"

// Going to borrow a lot of code from Vesselness 

#include "Data3D.h"
#include "ImageProcessing.h"

#include <fstream>
using namespace std; 

GLCamera cam;
GLViewer::Volumn *vObj; 
unsigned int sx = 0;
unsigned int sy = 0;
unsigned int sz = 0;

void reset_projection( int width, int height ) {
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix
	GLfloat maxVal = max( sx, max(sy, sz) ) * 1.0f;

	GLfloat ratio = (GLfloat)width / (GLfloat)height;

	glOrtho( -maxVal*ratio, maxVal*ratio, -maxVal, maxVal, -maxVal, maxVal);
	glMatrixMode(GL_MODELVIEW);
}

void reset_modelview(void) {
	cam.resetModelview( (GLfloat)sx, (GLfloat)sy, (GLfloat)sz );
}

///////////////////////////////////////////////////////////////////////////////
// default ctor
///////////////////////////////////////////////////////////////////////////////
ModelGL::ModelGL() : windowWidth(0), windowHeight(0), mouseLeftDown(false),
                     mouseRightDown(false), changeDrawMode(false), drawMode(0),
                     cameraAngleX(0), cameraAngleY(0), cameraDistance(5),
                     animateFlag(false), bgFlag(0), frameBuffer(0), bufferSize(0),
                     windowResized(false)
{
	Data3D<short> im_short;	
	im_short.load( "data/roi15.data" ); 
	IP::normalize( im_short, short(255) );
	Data3D<unsigned char> im_uchar = Data3D<unsigned char>( im_short );
	vObj = new GLViewer::Volumn( im_uchar.getMat().data, 
		im_uchar.SX(), im_uchar.SY(), im_uchar.SZ() );
}



///////////////////////////////////////////////////////////////////////////////
// destructor
///////////////////////////////////////////////////////////////////////////////
ModelGL::~ModelGL()
{
	
}



///////////////////////////////////////////////////////////////////////////////
// initialize OpenGL states and scene
///////////////////////////////////////////////////////////////////////////////
void ModelGL::init()
{
	glewInit();
	vObj->init();
	sx = max( sx, vObj->size_x() );
	sy = max( sy, vObj->size_y() );
	sz = max( sz, vObj->size_z() );

	reset_modelview();
}

///////////////////////////////////////////////////////////////////////////////
// set camera position and lookat direction
///////////////////////////////////////////////////////////////////////////////
void ModelGL::setCamera(float posX, float posY, float posZ, float targetX, float targetY, float targetZ)
{
    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();
    //gluLookAt(posX, posY, posZ, targetX, targetY, targetZ, 0, 1, 0); // eye(x,y,z), focal(x,y,z), up(x,y,z)
}



///////////////////////////////////////////////////////////////////////////////
// configure projection and viewport
///////////////////////////////////////////////////////////////////////////////
void ModelGL::setViewport(int w, int h)
{
    // assign the width/height of viewport
    windowWidth = w;
    windowHeight = h;
    //bufferSize = w * h * 4; // rgba
    //frameBuffer = new unsigned char [bufferSize];

    // set viewport to be the entire window
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);

    // set perspective viewing frustum
    float aspectRatio = (float)w / h;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0f, (float)(w)/h, 0.1f, 20.0f); // FOV, AspectRatio, NearClip, FarClip
	reset_projection( w, h );

    // switch to modelview matrix in order to set scene
    glMatrixMode(GL_MODELVIEW);
}



///////////////////////////////////////////////////////////////////////////////
// toggle to resize window
///////////////////////////////////////////////////////////////////////////////
void ModelGL::resizeWindow(int w, int h)
{
    // assign the width/height of viewport
    windowWidth = w;
    windowHeight = h;
    windowResized = true;
}



///////////////////////////////////////////////////////////////////////////////
// draw 2D/3D scene
///////////////////////////////////////////////////////////////////////////////
void ModelGL::draw()
{
	// clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // save the initial ModelView matrix before modifying ModelView matrix
    glPushMatrix();

    // tramsform camera
    glTranslatef(0, 0, cameraDistance);
    glRotatef(cameraAngleX, 1, 0, 0);   // pitch
    glRotatef(cameraAngleY, 0, 1, 0);   // heading

    // draw a triangle
    glBegin(GL_TRIANGLES);
        glColor3f(1, 0, 0);
        glVertex3f(3, -2, 0);
        glColor3f(0, 1, 0);
        glVertex3f(0, 2, 0);
        glColor3f(0, 0, 1);
        glVertex3f(-3, -2, 0);	
    glEnd();
	
	vObj->render(); 

    glPopMatrix();

	
}

///////////////////////////////////////////////////////////////////////////////
// rotate the camera
///////////////////////////////////////////////////////////////////////////////
void ModelGL::rotateCamera(int x, int y)
{
    if(mouseLeftDown)
    {
        cameraAngleY += (x - mouseX);
        cameraAngleX += (y - mouseY);
        mouseX = x;
        mouseY = y;
    }
}



///////////////////////////////////////////////////////////////////////////////
// zoom the camera
///////////////////////////////////////////////////////////////////////////////
void ModelGL::zoomCamera(int delta)
{
    if(mouseRightDown)
    {
        cameraDistance += (delta - mouseY) * 0.05f;
        mouseY = delta;
    }
}



///////////////////////////////////////////////////////////////////////////////
// change drawing mode
///////////////////////////////////////////////////////////////////////////////
void ModelGL::setDrawMode(int mode)
{
    if(drawMode != mode)
    {
        changeDrawMode = true;
        drawMode = mode;
    }
}



///////////////////////////////////////////////////////////////////////////////
// change background colour, the value should be between 0 and 1
///////////////////////////////////////////////////////////////////////////////
void ModelGL::setBackgroundRed(float value)
{
    if(bgColor[0] != value)
    {
        bgColor[0] = value;
        bgFlag = true;
    }
}
void ModelGL::setBackgroundGreen(float value)
{
    if(bgColor[1] != value)
    {
        bgColor[1] = value;
        bgFlag = true;
    }
}
void ModelGL::setBackgroundBlue(float value)
{
    if(bgColor[2] != value)
    {
        bgColor[2] = value;
        bgFlag = true;
    }
}
