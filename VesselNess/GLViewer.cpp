#include "GLViewer.h"

#include <iostream>
using namespace std;



#include <time.h>

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
	vector<bool> isDisplayObject;

	// Size of the data
	unsigned int sx = 0;
	unsigned int sy = 0;
	unsigned int sz = 0;

	/////////////////////////////////////////
	// Camera Controls by Mouse
	///////////////////////
	enum NavigationMode{
		none, 
		move_aside,
		move_forward,
		rotate
	} navigationMode = none; 
	// clicking position of the user
	int drag_x = 0;
	int drag_y = 0;
	// Rotation
	GLfloat	xrot = 0;              
	GLfloat	yrot = 0;
	GLfloat rotate_speed = 0.001f;
	// Rotation Axis
	GLfloat vec_y[3] = {0, 1, 0};
	GLfloat vec_x[3] = {1, 0, 0};
	// Translation
	GLfloat t[3] = { 0, 0, 0 };
	GLfloat translate_speed = 0.2f;
	
	int elapsedTick = 0;

	/////////////////////////////////////////
	// Initial Window Size
	///////////////////////
	int width = 1000;
	int height = 800;
	
	VideoSaver* videoSaver = NULL;

	bool isAxis = false;

	void rotate_scene(void){
		static int tick = GetTickCount();
		static int oldtick = GetTickCount(); 
		tick = GetTickCount(); 
		elapsedTick = tick - oldtick;
		oldtick = tick; 
		// update the rotation matrix as well as the rotation axis
		glTranslatef( t[0], t[1], t[2] );
		glRotatef( xrot * elapsedTick, vec_y[0], vec_y[1], vec_y[2] );
		rotate_axis( vec_y[0], vec_y[1], vec_y[2], 
			         vec_x[0], vec_x[1], vec_x[2],
			         vec_x[0], vec_x[1], vec_x[2], -xrot * elapsedTick );
		glRotatef( yrot * elapsedTick, vec_x[0], vec_x[1], vec_x[2] );
		rotate_axis( vec_x[0], vec_x[1], vec_x[2], 
			         vec_y[0], vec_y[1], vec_y[2],
			         vec_y[0], vec_y[1], vec_y[2], -yrot * elapsedTick );
		glTranslatef( -t[0], -t[1], -t[2] );
	}

	void draw_axis( void ) {
		glTranslatef( t[0], t[1], t[2] );
		// Draw Rotation Center with two axis
		glBegin(GL_LINES);
		glColor3f( 1.0, 0.0, 0.0 ); glVertex3i(  0,  0,  0 ); glVertex3f( vec_y[0]*10, vec_y[1]*10, vec_y[2]*10 );
		glColor3f( 0.0, 1.0, 0.0 ); glVertex3i(  0,  0,  0 ); glVertex3f( vec_x[0]*10, vec_x[1]*10, vec_x[2]*10 );
		glEnd();
		glTranslatef( -t[0], -t[1], -t[2] );
	}

	void render(void)									// Here's Where We Do All The Drawing
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear The Screen And The Depth Buffer

		// Viewport 1
		glViewport (0, 0, width/2, height);
		for( unsigned int i=0; i<obj.size(); i++ ) { 
			if( isDisplayObject[i] ) obj[i]->render();
		}
		if( isAxis ) draw_axis();

		// Viewport 2
		glViewport (width/2, 0, width/2, height);
		if( obj.size()>=1 ) obj[0]->render();
		if( isAxis ) draw_axis();

		rotate_scene();
		
		// saving frame buffer as video
		if( videoSaver ) videoSaver->saveBuffer();
		glutSwapBuffers();
	}

	void mouse_click(int button, int state, int x, int y) {
		if(button == GLUT_LEFT_BUTTON) { // mouse left button
			static int mouse_down_x;
			static int mouse_down_y;
			if(state == GLUT_DOWN) {
				navigationMode = rotate;
				drag_x = x;
				drag_y = y;
				mouse_down_x = x;
				mouse_down_y = y;
			} else if( state == GLUT_UP ){
				// stop tracking mouse move for rotating
				navigationMode = none; 
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
				navigationMode=move_aside; 
				drag_x = x;
				drag_y = y;
			} else {
				navigationMode=none;
			}
		} else if( button==GLUT_MIDDLE_BUTTON ) { // center button
			if( state == GLUT_DOWN ) {
				navigationMode = move_forward;
				drag_x = x;
				drag_y = y;
			} else {
				navigationMode = none;
			}
		} else if ( button==3 ) { // mouse wheel scrolling up
			// Zoom in
			glTranslatef( t[0], t[1], t[2] );
			glScalef( 1.01f, 1.01f, 1.01f );
			glTranslatef( -t[0], -t[1], -t[2] );
			translate_speed /= 1.01f; 
			rotate_speed  /= 1.01f; 
		} else if ( button==4 ) { // mouse wheel scrooling down 
			// Zoom out
			glTranslatef( t[0], t[1], t[2] );
			glScalef( 0.99f, 0.99f, 0.99f );
			glTranslatef( -t[0], -t[1], -t[2] );
			translate_speed /= 0.99f; 
			rotate_speed  /= 0.99f; 
		}
	}

	void mouse_move(int x, int y) {
		if( navigationMode == rotate ) {
			xrot = 1.0f*(x - drag_x) * rotate_speed;
			yrot = 1.0f*(y - drag_y) * rotate_speed;
			glutPostRedisplay();
		}
		if( navigationMode==move_aside ) {
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
		} else if( navigationMode==move_forward ) {
			GLfloat tx = (x - drag_x) * translate_speed;
			GLfloat ty = (y - drag_y) * translate_speed;
			GLfloat vec_z[3];
			vec_z[0] = vec_x[1]*vec_y[2] - vec_x[2]*vec_y[1]; 
			vec_z[1] = vec_x[2]*vec_y[0] - vec_x[0]*vec_y[2]; 
			vec_z[2] = vec_x[0]*vec_y[1] - vec_x[1]*vec_y[0]; 
			t[0] += (tx+ty) * vec_z[0];
			t[1] += (tx+ty) * vec_z[1];
			t[2] += (tx+ty) * vec_z[2];
		}
		// update mouse location
		drag_x = x;
		drag_y = y;
	}


	void reset_projection(void) {
		glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
		glLoadIdentity();									// Reset The Projection Matrix
		GLfloat maxVal = max( sx, max(sy, sz) ) * 0.9f;
		GLfloat ratio = (GLfloat)width/(GLfloat)height * 0.5f;
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

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity(); // clear the identity matrix.
		gluLookAt( 0, 0, 1, 0, 0, 0, 0, 1, 0 );

		// move to the center of the data
		glTranslatef(-t[0], -t[1], -t[2]); 

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
		if( key >= '1' && key <= '9' ){
			int index = key - '1';
			if( index < isDisplayObject.size() ) {
				isDisplayObject[index] = !isDisplayObject[index];
			}
		} 

		// additional key board control for objects
		if( key =='q' || key =='Q' && 0<obj.size() ) obj[0]->keyboard( '\t' );
		if( key =='w' || key =='W' && 1<obj.size() ) obj[1]->keyboard( '\t' );
		if( key =='e' || key =='E' && 2<obj.size() ) obj[2]->keyboard( '\t' );
		if( key =='r' || key =='R' && 3<obj.size() ) obj[3]->keyboard( '\t' );
		if( key =='t' || key =='T' && 4<obj.size() ) obj[4]->keyboard( '\t' );
		
		// TAB to toggle on/off the axis
		if( key =='\t' ) {
			isAxis = !isAxis;
		}

		switch (key) 
		{
		case ' ': 
			reset_projection();
			reset_modelview();
			break;
		case 27:
			cout << "Rednering done. Thanks you for using GLViewer. " << endl;
			exit(0);
		}
	}

	void go( vector<Object*> objects, VideoSaver* video )
	{
		obj = objects; 
		videoSaver = video;

		isDisplayObject.resize( objects.size(), false );
		isDisplayObject[0] = true;

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
			videoSaver->init(width,height);
		}

		cout << "Redenring Begin..." << endl;
		cout << "======================= Instructions =======================" << endl;
		cout << "   Mouse Controls: " << endl;
		cout << "       LEFT Button - Rotation " << endl;
		cout << "       RIGHT Button - Translation (aside) " << endl;
		cout << "       Middle Button - Translation (forward/backward) " << endl;
		cout << "   Keyboard Controls: " << endl;
		cout << "       TAB   - Toggle on/off Rotation Center " << endl;
		cout << "       SPACE - Reset Projection Matrix " << endl;
		cout << "       1,2,3 - Toggle on/off objects " << endl;
		cout << "       ESC   - Exit " << endl;

		glutMainLoop(); // No Code Will Be Executed After This Line
	}

}



