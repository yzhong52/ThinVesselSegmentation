#include "GLViewer.h"

#include <iostream>
using namespace std;

/////////////////////////////////////
// Glew Library 
// For Texture 3D and Blending_Ext
#include "gl\glew.h" 
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
	// objects that need to be render
	vector<Object*> obj;
	// Size of the data
	unsigned int sx = 0;
	unsigned int sy = 0;
	unsigned int sz = 0;

	/////////////////////////////////////////
	// Camera Controls by Mouse
	///////////////////////
	// clicking position of the user
	int drag_x = 0;
	int drag_y = 0;
	// Rotation
	GLfloat	xrot = 0;              
	GLfloat	yrot = 0;
	GLboolean isRotating = false;
	GLfloat rotate_speed = 0.05f;
	// Rotation Axis
	GLfloat vec_y[3] = {0, 1, 0};
	GLfloat vec_x[3] = {1, 0, 0};
	// Translation
	GLfloat t[3] = { 0, 0, 0 };
	GLboolean isTranslating = false;
	GLfloat translate_speed = 0.2f;
	
	
	
	/////////////////////////////////////////
	// Initial Window Size
	///////////////////////
	int width = 800;
	int height = 800;
	
	/////////////////////////////////////////
	// Additional Rendering Functions
	///////////////////////
	void (*after_render)(int,int) = NULL; // TODO: Remove this

	void render(void)									// Here's Where We Do All The Drawing
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear The Screen And The Depth Buffer

		glTranslatef( t[0], t[1], t[2] );
		if( after_render) { 
			// Not allow the user to retate the scene if we 
			// have after_render func, which is for saving videos
			glRotatef( 1.0f, 0.0f, 0.0f, 1.0f );
		} else {
			// Update Rotation Centre
			glRotatef( xrot, vec_y[0], vec_y[1], vec_y[2] );
			rotate_axis( vec_y[0], vec_y[1], vec_y[2], 
				vec_x[0], vec_x[1], vec_x[2],
				vec_x[0], vec_x[1], vec_x[2], -xrot );
			glRotatef( yrot, vec_x[0], vec_x[1], vec_x[2] );
			rotate_axis( vec_x[0], vec_x[1], vec_x[2], 
				vec_y[0], vec_y[1], vec_y[2],
				vec_y[0], vec_y[1], vec_y[2], -yrot );
			// Draw Rotation Center
			glBegin(GL_LINES);
			glColor3f( 1.0, 0.0, 0.0 ); glVertex3i(  0,  0,  0 ); glVertex3f( vec_y[0]*10, vec_y[1]*10, vec_y[2]*10 );
			glColor3f( 0.0, 1.0, 0.0 ); glVertex3i(  0,  0,  0 ); glVertex3f( vec_x[0]*10, vec_x[1]*10, vec_x[2]*10 );
			glEnd();
			glColor3f( 1.0, 1.0, 1.0 );
		}
		glTranslatef( -t[0], -t[1], -t[2] );


		for( unsigned int i=0; i<obj.size(); i++ ) { 
			if( obj[i]!=NULL ) obj[i]->render();
		}

		if( after_render ) after_render( width, height );

		glutSwapBuffers();
	}

	void mouse_click(int button, int state, int x, int y) {
		if(button == GLUT_LEFT_BUTTON) { // mouse left button
			static int mouse_down_x;
			static int mouse_down_y;
			if(state == GLUT_DOWN) {
				isRotating = true;
				drag_x = x;
				drag_y = y;
				mouse_down_x = x;
				mouse_down_y = y;
			} else if( state == GLUT_UP ){
				// stop tracking mouse move for rotating
				isRotating = false;
				// Stop the rotation immediately no matter what
				// if the user click and release the mouse at the
				// same point
				if( mouse_down_x==x && mouse_down_y==y ) {
					xrot = 0;
					yrot = 0;
				}
			}
		} else if(button == GLUT_RIGHT_BUTTON) { // mouse right button
			if( state == GLUT_DOWN ) {
				isTranslating = true;
				drag_x = x;
				drag_y = y;
			} else {
				isTranslating = false;
			}
		} else if ( button==3 ) { // mouse wheel scrolling up
			// Zoom in
			glTranslatef( t[0], t[1], t[2] );
			glScalef( 1.01f, 1.01f, 1.01f );
			glTranslatef( -t[0], -t[1], -t[2] );
		} else if ( button==4 ) { // mouse wheel scrooling down 
			// Zoom out
			glTranslatef( t[0], t[1], t[2] );
			glScalef( 0.99f, 0.99f, 0.99f );
			glTranslatef( -t[0], -t[1], -t[2] );
		}
	}

	void mouse_move(int x, int y) {
		if( isRotating ) {
			xrot = 1.0f*(x - drag_x) * rotate_speed;
			yrot = 1.0f*(y - drag_y) * rotate_speed;
			glutPostRedisplay();
		}
		if( isTranslating ) {
			GLfloat tx = -(x - drag_x) * translate_speed;
			GLfloat ty =  (y - drag_y) * translate_speed;
			glTranslatef( -tx*vec_x[0], -tx*vec_x[1], -tx*vec_x[2] );
			glTranslatef( -ty*vec_y[0], -ty*vec_y[1], -ty*vec_y[2] );
			t[0] += tx * vec_x[0];
			t[1] += tx * vec_x[1];
			t[2] += tx * vec_x[2];
			t[0] += ty * vec_y[0];
			t[1] += ty * vec_y[1];
			t[2] += ty * vec_y[2];
		}
		// update mouse location
		drag_x = x;
		drag_y = y;
	}


	void reset_projection(void) {
		glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
		glLoadIdentity();									// Reset The Projection Matrix
		float maxVal = max( sx, max(sy, sz) ) * 0.7f;
		GLfloat ratio = (GLfloat)width / (GLfloat)height;
		glOrtho( -1, 1, -1, 1, -1, 1);
		glScalef( 1.0f/(maxVal*ratio), 1.0f/maxVal, 1.0f/maxVal );
	}

	void reset_modelview(void) {
		t[0] = 0.5f*sx;
		t[1] = 0.5f*sy;
		t[2] = 0.5f*sz;
		vec_y[0] = 0; vec_y[1] = 1; vec_y[2] = 0;
		vec_x[0] = 1; vec_x[1] = 0; vec_x[2] = 0;
		xrot = 0;
		yrot = 0;

		glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
		glLoadIdentity();									// Reset The Projection Matrix

		GLfloat maxVal = max( sx, max(sy, sz) ) * 0.55f;
		GLfloat ratio = (GLfloat)width/(GLfloat)height;
		glOrtho( -1, 1, -1, 1, -1, 1);
		glScalef( 1.0f/(maxVal*ratio), 1.0f/maxVal, 1.0f/maxVal );

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity(); // clear the identity matrix.
		gluLookAt( 0, 0, 1,
			0, 0, 0, 
			0, 1, 0 );
		glTranslatef(-t[0], -t[1], -t[2]); // move to the center of the data

		if( after_render ) {
			// Initial Rotation (Do as you want );
			glTranslatef( t[0], t[1], t[2] );
			glRotatef( -90, vec_x[0], vec_x[1], vec_x[2] );
			glTranslatef( -t[0], -t[1], -t[2] );
		}

		glutPostRedisplay();
	}

	void reshape(int w, int h)
	{
		width = w; 
		height = (h==0) ? 1 : h;

		// Reset The Current Viewport
		glViewport( 0, 0, width, height );
		// Reset Projection
		reset_projection();
		// Reset Model View
		glMatrixMode(GL_MODELVIEW);
		glutPostRedisplay();
	}


	void keyboard(unsigned char key, int x, int y)
	{
		switch (key) 
		{
		case ' ': 
			reset_projection();
			reset_modelview();
			break;
		case 27:
			exit(0);
		}
	}

	void go( vector<Object*> objects )
	{
		int argc = 1;
		char* argv[1] = { NULL };
		glutInit( &argc, argv );
		glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB );
		glutInitWindowSize( width, height );
		glutInitWindowPosition( 100, 100 );
		glutCreateWindow( argv[0] );
		glewInit();

		obj = objects; 
		for( unsigned int i=0; i<obj.size(); i++ ) {
			obj[i]->init();
			sx = max( sx, obj[i]->size_x() );
			sy = max( sy, obj[i]->size_y() );
			sz = max( sz, obj[i]->size_z() );
		}
		
		///////////////////////////////////////////////
		// Register Recall Funtions
		///////////////
		glutReshapeFunc( reshape ); // Not allow the user to reshape the window if we 
		// have post_draw_func, which is for saving videos
		glutKeyboardFunc( keyboard );
		// register mouse fucntions
		glutMouseFunc( mouse_click );
		glutMotionFunc( mouse_move );
		// render func
		glutIdleFunc( render );
		glutDisplayFunc( render );

		reset_modelview();

		cout << "Redenring Begin!" << endl;
		glutMainLoop(); // No Code Will Be Executed After This Line
	}

}



