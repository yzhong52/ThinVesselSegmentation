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

#include "glew.h" // Yuchen
#pragma comment(lib, "glew32.lib") //Yuchen

#include <GL/gl.h>
#include <GL/glu.h>
#include "ModelGL.h"
#include "Bmp.h"
#include "smart_assert.h"

// Yuchen: there files are from in Porject VesselNess
// Borrowing to borrow a lot of code from Vesselness 
#include "Volumn.h" 
#include "Data3D.h"
#include "GLCamera.h" 
#include "ImageProcessing.h"

GLCamera cam;
GLViewer::Volumn *vObj; 
unsigned int sx = 0;
unsigned int sy = 0;
unsigned int sz = 0;

bool isUpdateTexture = false;
Data3D<short> im_short;	
Data3D<unsigned char> im_uchar;	

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
ModelGL::ModelGL() : windowWidth(0), windowHeight(0), windowResized(false)
{
	im_short.load( "data/roi15.data" ); 
	// convert data to unsigned char
	im_uchar.resize( im_short.get_size() ); 
	// copy data based on window center
	windowCenterMin = short(1<<15); 
	windowCenterMax = windowCenterMin-1;
	int window_size = max( 1, (int) windowCenterMax - (int)windowCenterMin );
	for( int i=0; i<im_uchar.get_size_total(); i++ ) {
		im_uchar.at(i) = 255 * ( (int) im_short.at(i) - (int)windowCenterMin ) / window_size; 
	}
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
// configure projection and viewport
///////////////////////////////////////////////////////////////////////////////
void ModelGL::setViewport()
{
	reset_projection( windowWidth, windowHeight );

	// set viewport to be the entire window
    glViewport(0, 0, (GLsizei)windowWidth, (GLsizei)windowHeight);

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
	// Yuchen: the draw function is in another process
	if( windowResized ) {
		setViewport();
		windowResized = false;
	}

	// clear buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	if( isUpdateTexture ) {
		Win::log( "Texture Updating" );
		vObj->update_texture();
		isUpdateTexture = false;
		Win::log( "Texture Updated" );
	}

	cam.scale_scene();
	cam.translate_scene();
	cam.rotate_scene();
	vObj->render(); 
	cam.draw_axis();
	cam.translate_scene_reverse();
	cam.scale_scene_reverse();
}


// Mouse Control Message
////////////////////////////////////////////////
// Mouse Button Down
////////////////////////////////////////////////
void ModelGL::mouseDown_RightButton( int x, int y )
{
	cam.setNavigationMode( GLCamera::MoveAside );
	mouse_pos_x = x;
	mouse_pos_y = y;
}

void ModelGL::mouseDown_LeftButton( int x, int y )
{
	cam.setNavigationMode( GLCamera::Rotate );
	mouse_pos_x = x;
	mouse_pos_y = y;
	mouse_down_x = x;
	mouse_down_y = y;
}

void ModelGL::mouseDown_MiddleButton( int x, int y )
{
	cam.setNavigationMode( GLCamera::None );
	mouse_pos_x = x; 
	mouse_pos_y = y; 
}

////////////////////////////////////////////////
// Mouse Button Up
////////////////////////////////////////////////
void ModelGL::mouseUp_LeftButton( int x, int y )
{
	// stop tracking mouse move for rotating
	cam.setNavigationMode( GLCamera::None );
	// Stop the rotation immediately no matter what
	// if the user click and release the mouse at the
	// same point
	if( mouse_down_x==x && mouse_down_y==y ) {
		cam.setRotation( 0, 0 ); // stop rotation
	}
}

void ModelGL::mouseUp_MiddleButton( int x, int y )
{
	cam.setNavigationMode( GLCamera::None );
}

void ModelGL::mouseUp_RightButton( int x, int y )
{
	cam.setNavigationMode( GLCamera::None );
}


void ModelGL::mouseMove_LeftButton( int x, int y ){
	if( cam.getNavigationMode() == GLCamera::Rotate ) {
		cam.setRotation( 1.0f*(x - mouse_pos_x), 1.0f*(y - mouse_pos_y) );
	}
	mouse_pos_x = x; // update mouse location
	mouse_pos_y = y; // update mouse location
}

void ModelGL::mouseMove_RightButton( int x, int y ){
	if( cam.getNavigationMode() == GLCamera::MoveAside ) {
		cam.translate_aside( mouse_pos_x-x, mouse_pos_y-y );	
	}
	mouse_pos_x = x; // update mouse location
	mouse_pos_y = y; // update mouse location
}

void ModelGL::mouseMove_MiddleButton( int x, int y ){
	if( cam.getNavigationMode()==GLCamera::MoveForward ) {
		cam.translate_forward( x - mouse_pos_x, y - mouse_pos_y );
	}
	mouse_pos_x = x; // update mouse location
	mouse_pos_y = y; // update mouse location
}

void ModelGL::mouseWheel_Up( void ){
	cam.zoomIn(); 
}

void ModelGL::mouseWheel_Down( void ){
	cam.zoomOut(); 
}


void ModelGL::updateWindowCenterMin( int position )
{
	windowCenterMin = position; 
	windowCenterUpdateData(); 
}

void ModelGL::updateWindowCenterMax( int position )
{
	windowCenterMax = position; 
	windowCenterUpdateData(); 
}


void ModelGL::updateWindowCenter( int min, int max ){
	windowCenterMin = min; 
	windowCenterMax = max; 
	windowCenterUpdateData(); 
}

void ModelGL::windowCenterUpdateData() {
	Win::log( "Updating Window Center [%d, %d]", windowCenterMin, windowCenterMax );
	
	int window_size = max( 1, windowCenterMax-windowCenterMin );
	for( int i=0; i<im_uchar.get_size_total(); i++ ) {
		if( im_short.at(i) < windowCenterMin ) { 
			im_uchar.at(i) = 0; 
		} else if ( im_short.at(i) > windowCenterMax ) { 
			im_uchar.at(i) = 255; 
		} else {
			im_uchar.at(i) = 255 * ( (int) im_short.at(i) - (int) windowCenterMin ) / window_size; 
		}
	}
	vObj->update_data( im_uchar.getMat().data ); 
	isUpdateTexture = true;
}
