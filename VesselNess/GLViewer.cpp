#include "GLViewer.h"
#include "GLCamera.h"

#include <iostream>
using namespace std;

#include <time.h>

namespace GLViewer
{
	// objects that need to be render
	vector<Object*> obj;
	const int maxNumViewports = 1; 
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
	int drag_x = 0;
	int drag_y = 0;

	/////////////////////////////////////////
	// Initial Window Size
	///////////////////////
	int width = 512 * numViewports;
	int height = 512;
	
	VideoSaver* videoSaver = NULL;

	bool isAxis = false;
	

	void render(void)									// Here's Where We Do All The Drawing
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear The Screen And The Depth Buffer

		// rending two viewports
		for( int i=0; i<numViewports; i++ ) {
			// For viewport i
			glViewport (i*width/numViewports, 0, width/numViewports, height);
			for( unsigned int j=0; j<obj.size(); j++ ) { 
				if( isDisplayObject[i][j] ) obj[j]->render();
			}
			if( isAxis ) cam.draw_axis();
		}

		cam.rotate_scene();
		
		// saving frame buffer as video
		if( videoSaver ) videoSaver->saveBuffer();
		glutSwapBuffers();
	}

	void mouse_click(int button, int state, int x, int y) {
		if(button == GLUT_LEFT_BUTTON) { // mouse left button
			static int mouse_down_x;
			static int mouse_down_y;
			if(state == GLUT_DOWN) {
				cam.setNavigationMode( GLCamera::Rotate );
				drag_x = x;
				drag_y = y;
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
				drag_x = x;
				drag_y = y;
			} else {
				cam.setNavigationMode( GLCamera::None );
			}
		} else if( button==GLUT_MIDDLE_BUTTON ) { // center button
			if( state == GLUT_DOWN ) {
				cam.setNavigationMode( GLCamera::MoveForward );
				drag_x = x;
				drag_y = y;
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
			cam.setRotation( 1.0f*(x - drag_x), 1.0f*(y - drag_y) );
			glutPostRedisplay();
		} else  if( cam.getNavigationMode() == GLCamera::MoveAside ) {
			cam.translate_aside( x - drag_x, y - drag_y );
		} else if( cam.getNavigationMode()==GLCamera::MoveForward ) {
			cam.translate_forward( x - drag_x, y - drag_y );
		}
		drag_x = x; // update mouse location
		drag_y = y; // update mouse location
	}


	void reset_projection(void) {
		glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
		glLoadIdentity();									// Reset The Projection Matrix
		GLfloat maxVal = max( sx, max(sy, sz) ) * 1.0f;
		
		GLfloat ratio = (GLfloat)width / (GLfloat)height / numViewports;
		
		glOrtho( -maxVal*ratio, maxVal*ratio, -maxVal, maxVal, -maxVal, maxVal);
		glMatrixMode(GL_MODELVIEW);
	}

	void reset_modelview(void) {
		cam.resetModelview( (GLfloat)sx, (GLfloat)sy, (GLfloat)sz );
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
			int index = key - '1';
			if( index < isDisplayObject[numViewports-1].size() ) {
				isDisplayObject[numViewports-1][index] = !isDisplayObject[numViewports-1][index];
			}
		} 

		// additional key board control for objects
		if( key =='q' || key =='Q' && 0<obj.size() ) obj[0]->keyboard( '\t' );
		if( key =='w' || key =='W' && 1<obj.size() ) obj[1]->keyboard( '\t' );
		if( key =='e' || key =='E' && 2<obj.size() ) obj[2]->keyboard( '\t' );
		if( key =='r' || key =='R' && 3<obj.size() ) obj[3]->keyboard( '\t' );
		if( key =='t' || key =='T' && 4<obj.size() ) obj[4]->keyboard( '\t' );
		
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
		case 27:
			cout << "Rednering done. Thanks you for using GLViewer. " << endl;
			exit(0);
			break;
		}
	}

	void go( vector<Object*> objects, VideoSaver* video )
	{
		obj = objects; 
		videoSaver = video;
		for( int i=0; i<maxNumViewports; i++ ){ 
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
			glTranslatef( 0.5f*sx, 0.5f*sy, 0.5f*sx );
			glRotatef( -90, 1, 0, 0 );
			glTranslatef( -0.5f*sx, -0.5f*sy, -0.5f*sx );
			videoSaver->init(width, height);
		}

		cout << "Redenring Begin..." << endl;
		cout << "======================= Instructions =======================" << endl;
		cout << "   Mouse Controls: " << endl;
		cout << "       LEFT Button - Rotation " << endl;
		cout << "       RIGHT Button - Translation (aside) " << endl;
		cout << "       Middle Button - Translation (forward/backward) " << endl;
		cout << "   Keyboard Controls: " << endl;
		cout << "       TAB   - Toggle on/off The Second Viewport" << endl;
		cout << "       a     - Toggle on/off Rotation Center " << endl;
		cout << "       SPACE - Reset Projection Matrix " << endl;
		cout << "       1,2,3 - Toggle on/off objects " << endl;
		cout << "       ESC   - Exit " << endl;

		glutMainLoop(); // No Code Will Be Executed After This Line
	}

}



