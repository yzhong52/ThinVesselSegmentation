#include "GLViewer.h"

#include <iostream>
using namespace std;

/////////////////////////////////////
// Glew Library 
// For Texture 3D and Blending_Ext
#include "glew.h" 
#pragma comment(lib, "glew32.lib")

/////////////////////////////////////
// OpenGL Library
#include <windows.h>		// Header File For Windows
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library

/////////////////////////////////////
// Glut Library
#include "GLUT\glut.h"
#pragma comment(lib, "freeglut.lib")

void rotate_axis( 
	float u, float v, float w,        /*Axis*/
	float x, float y, float z,        /*The Point That We Want to Roate */
	float& nx, float& ny, float& nz,  /*Result*/
	float degree ) 
{
	float A = degree * 3.14159265f / 180.0f;
	float c = cos(A);
	float s = sin(A);
	float C = 1.0f - c;
	
	if( abs(c) > 0.999 ) {
		nx = x;
		ny = y;
		nz = z;
	}

	// Reference: http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
	float Q[3][3];
	Q[0][0] = u * u * C + c;
	Q[0][1] = v * u * C + w * s;
	Q[0][2] = w * u * C - v * s;

	Q[1][0] = v * u * C - w * s;
	Q[1][1] = v * v * C + c;
	Q[1][2] = w * v * C + u * s;

	Q[2][0] = u * w * C + v * s;
	Q[2][1] = w * v * C - u * s;
	Q[2][2] = w * w * C + c;

	nx = x * Q[0][0] + y * Q[1][0] + z * Q[2][0];
	ny = x * Q[0][1] + y * Q[1][1] + z * Q[2][1];
	nz = x * Q[0][2] + y * Q[1][2] + z * Q[2][2];
}


namespace GLViewer
{
	/////////////////////////////////////////
	// Camera Controls by Mouse
	// Rotation
	GLfloat	xrot = 0;              
	GLfloat	yrot = 0;
	int drag_x = 0;   // clicking position of the user
	int drag_y = 0;   // clicking position of the user
	GLboolean isRotating = false;  // is dragging by user
	// Translation
	GLfloat tx = 0;
	GLfloat ty = 0;
	// Zooming 
	GLfloat zoom = 1;
	
	GLuint texture;     // Texture

	/////////////////////////////////////////
	// Data
	int sx = 0;
	int sy = 0;
	int sz = 0;
	unsigned char* data = NULL;

	// size of the window
	int width = 800;
	int height = 800;

	void (*extra_render)() = NULL;
	void (*after_render)(int,int) = NULL;

	GLfloat vec_y[3] = {0, 1, 0};
	GLfloat vec_x[3] = {1, 0, 0};

	float rotate_speed = 0.05f;

	void render(void)									// Here's Where We Do All The Drawing
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear The Screen And The Depth Buffer

		glTranslatef( 0.5f*sx, 0.5f*sy, 0.5f*sz );
		
		// Not allow the user to retate the scene if we 
		// have after_render func, which is for saving videos
		if( after_render) { 
			glRotatef( 1.0f, 0.0f, 0.0f, 1.0f );
		} else {
			glRotatef( xrot, vec_y[0], vec_y[1], vec_y[2] );
			rotate_axis( vec_y[0], vec_y[1], vec_y[2], 
					 	 vec_x[0], vec_x[1], vec_x[2],
						 vec_x[0], vec_x[1], vec_x[2], -xrot );
			glRotatef( yrot, vec_x[0], vec_x[1], vec_x[2] );
			rotate_axis( vec_x[0], vec_x[1], vec_x[2], 
					 	 vec_y[0], vec_y[1], vec_y[2],
						 vec_y[0], vec_y[1], vec_y[2], -yrot );

			glBegin(GL_LINES);
			glColor3f( 1.0, 0.0, 0.0 );
			glVertex3i(  0,  0,  0 ); glVertex3f( vec_y[0]*20, vec_y[1]*20, vec_y[2]*20 );
			glColor3f( 0.0, 1.0, 0.0 );
			glVertex3i(  0,  0,  0 ); glVertex3f( vec_x[0]*20, vec_x[1]*20, vec_x[2]*20 );
			glEnd();
		}
		glTranslatef( -0.5f*sx, -0.5f*sy, -0.5f*sz );
		
		// Allow User to draw additional objects on the scene
		if( extra_render != NULL ) extra_render();

		glBindTexture(GL_TEXTURE_3D, texture);
		glBegin(GL_QUADS);
		for( int i=0; i<sz+1; i++ ) {
			glTexCoord3f( 0, 0, 1.0f*i/sz ); glVertex3i(  0,  0, i );
			glTexCoord3f( 1, 0, 1.0f*i/sz ); glVertex3i( sx,  0, i );
			glTexCoord3f( 1, 1, 1.0f*i/sz ); glVertex3i( sx, sy, i );
			glTexCoord3f( 0, 1, 1.0f*i/sz ); glVertex3i(  0, sy, i );
		}
		for( int i=0; i<sy+1; i++ ) {
			glTexCoord3f( 0, 1.0f*i/sy, 0 ); glVertex3i(  0, i,  0 );
			glTexCoord3f( 1, 1.0f*i/sy, 0 ); glVertex3i( sx, i,  0 );
			glTexCoord3f( 1, 1.0f*i/sy, 1 ); glVertex3i( sx, i, sz );
			glTexCoord3f( 0, 1.0f*i/sy, 1 ); glVertex3i(  0, i, sz );
		}
		for( int i=0; i<sx+1; i++ ) {
			glTexCoord3f( 1.0f*i/sx, 0, 0 ); glVertex3i( i,  0, 0 );
			glTexCoord3f( 1.0f*i/sx, 1, 0 ); glVertex3i( i, sy, 0 );
			glTexCoord3f( 1.0f*i/sx, 1, 1 ); glVertex3i( i, sy, sz );
			glTexCoord3f( 1.0f*i/sx, 0, 1 ); glVertex3i( i,  0, sz );
		}
		glEnd();

		glBindTexture( GL_TEXTURE_3D, NULL );
		
		if( after_render ) after_render( width, height );
		
		glutSwapBuffers();
	}

	void mouse_click(int button, int state, int x, int y) {
		if(button == GLUT_LEFT_BUTTON) {
			if(state == GLUT_DOWN) {
				isRotating = true;
				drag_x = x;
				drag_y = y;
			} else {
				isRotating = false;
			}
		} else if ( button==3 ) {
			// mouse wheel scrolling up
			zoom *= 1.05f;
		} else if ( button==4 ) {
			// mouse wheel scrooling down 
			zoom *= 0.95f;
		}
	}

	void mouse_move(int x, int y) {
		if(isRotating) {
			xrot = 1.0f*(x - drag_x);
			yrot = 1.0f*(y - drag_y);
			if( abs(xrot) <= 1 ) xrot = 0;
			if( abs(yrot) <= 1 ) yrot = 0;
			xrot *= rotate_speed;
			yrot *= rotate_speed;

			drag_x = x;
			drag_y = y;
			glutPostRedisplay();
		}
	}
	
	void reset_modelview(void) {
		glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
		glLoadIdentity();									// Reset The Projection Matrix

		int maxVal = max( sx, max(sy, sz) );
		GLfloat ratio = (GLfloat)width/(GLfloat)height;
		glOrtho( -1, 1, -1, 1, -1, 1);
		glScalef( 1.0f/(maxVal*ratio), 1.0f/maxVal, 1.0f/maxVal );

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity(); // clear the identity matrix.
		gluLookAt( 0, 0, 1,
			       0, 0, 0, 
				   0, 1, 0 );
		glTranslatef(-0.5f*sx, -0.5f*sy, -0.5f*sz); // move to the center of the data
		
		//glTranslatef( 0.5f*sx, 0.5f*sy, 0.5f*sz );
		//glRotatef( -90, 1.0f, 0.0f, 0.0f );
		//glTranslatef( -0.5f*sx, -0.5f*sy, -0.5f*sz );

		glutPostRedisplay();
	}

	void reshape(int w, int h)
	{
		// Yuchen: Code Modified From Nehe
		// Calculate The Aspect Ratio Of The Window
		width = w; 
		height = (h==0) ? 1 : h;

		glViewport(0,0,w,h);						// Reset The Current Viewport
		
		glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
		glLoadIdentity();									// Reset The Projection Matrix
		float maxVal = max( sx, max(sy, sz) ) * 0.7f;
		GLfloat ratio = (GLfloat)w/(GLfloat)h;
		glOrtho( -1, 1, -1, 1, -1, 1);
		glScalef( 1.0f/(maxVal*ratio), 1.0f/maxVal, 1.0f/maxVal );
		
		glMatrixMode(GL_MODELVIEW);
		glutPostRedisplay();
	}


	void keyboard(unsigned char key, int x, int y)
	{
		switch (key) 
		{
		case ' ': 
			reset_modelview();
			break;
		case 27:
			exit(0);
		}
	}

	void MIP( unsigned char* im_data, int im_x, int im_y, int im_z, 
		void (*pre_draw_func)(void),
		void (*post_draw_func)(int,int) )
	{
		sx = (im_x%2==0) ? im_x : im_x+1;
		sy = (im_y%2==0) ? im_y : im_y+1;
		sz = (im_z%2==0) ? im_z : im_z+1;
		if( sx==im_x && sy==im_y && sz==im_z ) {
			data = im_data;
		} else {
			data = new unsigned char[ sx*sy*sz ];
			for( int z=0;z<im_z;z++ ) for( int y=0;y<im_y;y++ ) for( int x=0;x<im_x;x++ ) {
				data[ z*sy*sx + y*sx + x] = im_data[ z*im_y*im_x + y*im_x + x];
			}
		}

		extra_render = pre_draw_func;
		after_render = post_draw_func;

		int argc = 1;
		char* argv[1] = { NULL };
		glutInit( &argc, argv );
		glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB );
		glutInitWindowSize( width, height );
		glutInitWindowPosition( 100, 100 );
		glutCreateWindow( argv[0] );
		glewInit();
		
		//////////////////////////////////////
		// Set up OpenGL
		glEnable( GL_TEXTURE_3D ); // Enable Texture Mapping
		glBlendFunc(GL_ONE, GL_ONE);
		glEnable(GL_BLEND);
		glBlendEquationEXT( GL_MAX_EXT ); // Enable Blending For Maximum Intensity Projection

		// Creating Textures
		glGenTextures(1, &texture); // Create The Texture
		glBindTexture(GL_TEXTURE_3D, texture);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB, 
			sx, sy, sz, 0,
			GL_LUMINANCE, GL_UNSIGNED_BYTE, data );

		// Use GL_NEAREST to see the voxels
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ); 
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		// Sets the wrap parameter for texture coordinate
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE  ); 
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE  );
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE  );
		
		glEnable( GL_POLYGON_SMOOTH_HINT );
		glHint (GL_POLYGON_SMOOTH_HINT, GL_NICEST);
		glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
		

		// Register Recall Funtions
		if( !post_draw_func ) glutReshapeFunc( reshape ); // Not allow the user to reshape the window if we 
		                                                  // have post_draw_func, which is for saving videos
		glutKeyboardFunc( keyboard );
		// register mouse fucntions
		glutMouseFunc( mouse_click );
		glutMotionFunc( mouse_move );
		// render func
		glutIdleFunc( render );
		glutDisplayFunc( render );

		// Release Texture Data
		if( data != im_data ) delete[] data;

		reset_modelview();
		glutMainLoop(); // No Code Will Be Executed After This Line
	}
}

