#include "GLViewer.h"
#include "GLCamera.h"

#include <iostream>
using namespace std;

#include <time.h>

/////////////////////////////////////
// Glut Library
#include "GL\freeglut.h"
#pragma comment(lib, "freeglut.lib")

namespace GLViewer
{
	// objects that need to be render
	vector<Object*> obj;
	const int maxNumViewports = 2; 
	int numViewports = maxNumViewports;
	vector<bool> isDisplayObject[maxNumViewports];
	
	// Size of the data
	unsigned int sx = 0;
	unsigned int sy = 0;
	unsigned int sz = 0;

	/////////////////////////////////////////
	// Camera Controls by Mouse
	///////////////////////
	GLCamera cam; 
	int mouse_pos_x = 0;
	int mouse_pos_y = 0;

	/////////////////////////////////////////
	// Initial Window Size
	///////////////////////
	int width = 1280 /2 * numViewports;
	int height = 720;
	
	VideoSaver* videoSaver = NULL;

	bool isAxis = false;
	
	bool isSaveFrame = false; 

	void render(void)									// Here's Where We Do All The Drawing
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear The Screen And The Depth Buffer
		
		cam.pushMatrix();
		// rending viewports
		for( int i=0; i<numViewports; i++ ) {
			// For viewport i
			glViewport (i*width/numViewports, 0, width/numViewports, height);
			for( unsigned int j=0; j<obj.size(); j++ ) { 
				if( isDisplayObject[i][j] ) obj[j]->render();
			}
			if( isAxis ) {
				// draw rotation matrix
				cam.draw_axis();
			}
		}
		cam.popMatrix(); 
		cam.rotate_scene(); 
		
		// saving frame buffer as video
		if( videoSaver ) {
			static bool autoRotate = true;
			if( autoRotate ) {
				cam.rotate_y(1); // rotate the camera for 1 degree 
			}
			videoSaver->saveBuffer();
		}
		if( isSaveFrame ) {
			videoSaver->takeScreenShot( width, height );
			isSaveFrame = false; 
		}

		glutSwapBuffers();
	}

	void mouse_click(int button, int state, int x, int y) {
		if(button == GLUT_LEFT_BUTTON) { // mouse left button
			static int mouse_down_x;
			static int mouse_down_y;
			if(state == GLUT_DOWN) {
				cam.setNavigationMode( GLCamera::Rotate );
				mouse_pos_x = x;
				mouse_pos_y = y;
				mouse_down_x = x;
				mouse_down_y = y;
			} else if( state == GLUT_UP ){
				// stop tracking mouse move for rotating
				cam.setNavigationMode( GLCamera::None );
				// Stop the rotation immediately no matter what
				// if the user click and release the mouse at the
				// same point
				if( mouse_down_x==x && mouse_down_y==y ) {
					cam.setRotation( 0, 0 ); // stop rotation
				}
			}
		} else if(button == GLUT_RIGHT_BUTTON) { // mouse right button
			if( state == GLUT_DOWN ) {
				cam.setNavigationMode( GLCamera::MoveAside );
				mouse_pos_x = x;
				mouse_pos_y = y;
			} else {
				cam.setNavigationMode( GLCamera::None );
			}
		} else if( button==GLUT_MIDDLE_BUTTON ) { // center button
			if( state == GLUT_DOWN ) {
				cam.setNavigationMode( GLCamera::MoveForward );
				mouse_pos_x = x;
				mouse_pos_y = y;
			} else {
				cam.setNavigationMode( GLCamera::None );
			}
		} else if ( button==3 ) { // mouse wheel scrolling up
			cam.zoomIn();
		} else if ( button==4 ) { // mouse wheel scrooling down 
			cam.zoomOut();
		}
	}

	// the mouse_move function will only be called when at least one button of the mouse id down
	void mouse_move(int x, int y) {
		if( cam.getNavigationMode() == GLCamera::Rotate ) {
			cam.setRotation( 1.0f*(x - mouse_pos_x), 1.0f*(y - mouse_pos_y) );
			glutPostRedisplay();
		} else if( cam.getNavigationMode() == GLCamera::MoveAside ) {
			cam.translate_aside( x - mouse_pos_x, y - mouse_pos_y );
		} else if( cam.getNavigationMode()==GLCamera::MoveForward ) {
			cam.translate_forward( x - mouse_pos_x, y - mouse_pos_y );
		}
		mouse_pos_x = x; // update mouse location
		mouse_pos_y = y; // update mouse location
	}


	void reset_projection(void) {
		glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
		glLoadIdentity();									// Reset The Projection Matrix
		GLfloat maxVal = max( sx, max(sy, sz) ) * 0.8f;
		
		GLfloat ratio = (GLfloat)width / (GLfloat)height / numViewports;
		
		glOrtho( -maxVal*ratio, maxVal*ratio, -maxVal, maxVal, -maxVal, maxVal);
		glMatrixMode(GL_MODELVIEW);
	}

	void reset_modelview(void) {
		cam.resetModelview( (GLfloat)sx, (GLfloat)sy, (GLfloat)sz );
		glutPostRedisplay(); 
	}

	void reshape(int w, int h)
	{
		width = max(50, w); 
		height = max(50, h);

		reset_projection(); // Reset Projection

		glutPostRedisplay();
	}


	void keyboard(unsigned char key, int x, int y)
	{
		if( key >= '1' && key <= '9' ){
			unsigned int index = key - '1';
			if( index >= isDisplayObject[numViewports-1].size() )  return;

			if (glutGetModifiers() == GLUT_ACTIVE_ALT ) {
				obj[index]->keyboard( '\t' );
			} else {
				isDisplayObject[numViewports-1][index] = !isDisplayObject[numViewports-1][index];
			}
		} 

		switch (key) 
		{
		case ' ': 
			reset_projection();
			reset_modelview();
			break;
		case 'a': 
			// toggle on/off the axis
			isAxis = !isAxis;
			break;
		case '\t':
			numViewports = (numViewports+1)%maxNumViewports + 1; 
			reset_projection();
			break;
		case 's': case 'S': 
			isSaveFrame = true; 
			break;
		case 27:
			cout << "Rednering done. Thanks you for using GLViewer. " << endl;
			exit(0);
			break;
		}
	}

	void go( vector<Object*> objects, VideoSaver* video )
	{
		if( objects.size()==0 ) {
			std::cerr << "No Objects For Rendering... " << std::endl;
			return;
		}

		obj = objects; 
		videoSaver = video;
		for( unsigned int i=0; i<maxNumViewports; i++ ){ 
			isDisplayObject[i].resize( objects.size(), false );
			if(i<objects.size()) { // put the i-th object in the i-th viewport
				isDisplayObject[i][i] = true;
			} else {
				isDisplayObject[i][0] = true;
			}
		}

		///////////////////////////////////////////////
		// glut Initializaitons
		///////////////
		int argc = 1;
		char* argv[1] = { NULL };
		glutInit( &argc, argv );
		glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB );
		glutInitWindowSize( width, height );
		glutInitWindowPosition( 100, 100 );
		glutCreateWindow( argv[0] );
		glewInit();
		// Register Recall Funtions
		glutReshapeFunc( reshape );
		// have post_draw_func, which is for saving videos
		glutKeyboardFunc( keyboard );
		// register mouse fucntions
		glutMouseFunc( mouse_click );
		glutMotionFunc( mouse_move );
		// render func
		glutIdleFunc( render );
		glutDisplayFunc( render );


		// Global Inits - Antialiasing
		glEnable (GL_LINE_SMOOTH);
		glHint (GL_LINE_SMOOTH_HINT, GL_NICEST );
		// The order of the following settings do matters
		// Setting up the size of the scence
		for( unsigned int i=0; i<obj.size(); i++ ) {
			obj[i]->init(); // additionol init settings by objects
			sx = max( sx, obj[i]->size_x() );
			sy = max( sy, obj[i]->size_y() );
			sz = max( sz, obj[i]->size_z() );
		}

		// reset the modelview and projection
		reset_projection();
		reset_modelview();

		// setting up for video captures if there is any
		if( videoSaver ) {
			// Initial Rotation (Do as you want ); Now it is faciton the x-y plane
			cam.rotate_x(-90); // rotate around x-axis for 90 degrees
			videoSaver->init(width, height);
		}
		
		cout << "Redenring Begin..." << endl;
		cout << "======================= Instructions =======================" << endl;
		cout << "   Mouse Controls: " << endl;
		cout << "       LEFT Button       - Rotation " << endl;
		cout << "       RIGHT Button      - Translation (aside) " << endl;
		cout << "       Middle Button     - Translation (forward/backward) " << endl;
		cout << "   Keyboard Controls: " << endl;
		cout << "       TAB               - Toggle on/off The Second Viewport" << endl;
		cout << "       a                 - Toggle on/off Rotation Center " << endl;
		cout << "       SPACE             - Reset Projection Matrix " << endl;
		cout << "       1,2,3...          - Toggle on/off objects " << endl;
		cout << "       ALT + 1,2,3...  - Change Rendering Mode for a object " << endl;
		cout << "       ESC               - Exit " << endl;

		glutMainLoop(); // No Code Will Be Executed After This Line
	}

}



