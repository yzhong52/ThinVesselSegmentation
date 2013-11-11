#include "GLViewer.h"

// Yuchen: Adding glut for OpenCV
#include "..\VesselNess\dependencies\freeglut 2.8.1\include\GL\glut.h"
#include "Edge_Graph.h"
#include <vector>

namespace GLViewer
{
		// global data pointer
		const Edge_Graph<Edge_Ext>* ptrGraph = NULL;
		const std::vector<Line>* ptrLines = NULL;

		// setting parameters
		float threshold = 0.0f;
		// Some global variables
		// zooming
		float zoom = 1.0f; 
		// For Camera Rotation
		double camera_angle_h = 0; 
		double camera_angle_v = 0; 
		int drag_x_origin;
		int drag_y_origin;
		int dragging = 0;
		// id for the window that we create
		int window_id;

		void renderDirFunc( void )
		{
			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

			//// calculate the center of the scene
			std::vector<Line>::const_iterator it = ptrLines->begin();
			Vec3f min_pos( it->p1 );
			Vec3f max_pos( it->p1 );

			for( it=ptrLines->begin(); it < ptrLines->end(); it++ ) {
				const Line& l = *it; 
				min_pos.x = std::min( it->p1.x, min_pos.x );
				min_pos.y = std::min( it->p1.y, min_pos.y );
				min_pos.z = std::min( it->p1.z, min_pos.z );
				min_pos.x = std::min( it->p2.x, min_pos.x );
				min_pos.y = std::min( it->p2.y, min_pos.y );
				min_pos.z = std::min( it->p2.z, min_pos.z );

				max_pos.x = std::max( it->p1.x, max_pos.x );
				max_pos.y = std::max( it->p1.y, max_pos.y );
				max_pos.z = std::max( it->p1.z, max_pos.z );
				max_pos.x = std::max( it->p2.x, max_pos.x );
				max_pos.y = std::max( it->p2.y, max_pos.y );
				max_pos.z = std::max( it->p2.z, max_pos.z );
			}
			Vec3f center = (max_pos + min_pos) * 0.5f; 

			//// camera
			glLoadIdentity();
			glOrtho( -center.x, center.x, -center.y, center.y, -100000.0f, 100000.0f);
			gluLookAt( /*orgin*/ 0.0, 0.0, 1.0, /*look at*/ 0.0, 0.0, 0.0, /*up vector*/0.0, 1.0, 0.0);

			glRotated( camera_angle_v, 1.0, 0.0, 0.0);
			glRotated( camera_angle_h, 0.0, 1.0, 0.0);

			//// 
			glColor3f(1,0,0);
			glBegin( GL_LINES );
			for( it=ptrLines->begin(); it < ptrLines->end(); it++ ) {
				const Line& l = *it; 
				glVertex3f( zoom*(l.p1.x-center.x), zoom*(l.p1.y-center.y), zoom*(l.p1.z-center.z) );
				glVertex3f( zoom*(l.p2.x-center.x), zoom*(l.p2.y-center.y), zoom*(l.p2.z-center.z) );
			}
			glEnd();

			glColor3f(0,1,1);
			glBegin( GL_LINES );
			const Edge_Ext* pt_edge = &(ptrGraph->get_edges().top());
			for( unsigned int i=0; i<ptrGraph->get_num_edges(); i++ ) {
				glVertex3f( 
					zoom*(pt_edge->p1.x-center.x), 
					zoom*(pt_edge->p1.y-center.y), 
					zoom*(pt_edge->p1.z-center.z) );
				glVertex3f( 
					zoom*(pt_edge->p2.x-center.x), 
					zoom*(pt_edge->p2.y-center.y), 
					zoom*(pt_edge->p2.z-center.z) );
				pt_edge++;
			}
			glEnd();
			
			glutSwapBuffers();
		}

		void mouse_click(int button, int state, int x, int y) {
			if(button == GLUT_LEFT_BUTTON) {
				if(state == GLUT_DOWN) {
					dragging = 1;
					drag_x_origin = x;
					drag_y_origin = y;
				} else {
					dragging = 0;
				}
			} else if ( button==3 ) {
				// mouse wheel scrolling up
				zoom *= 1.1f;
			} else if ( button==4 ) {
				// mouse wheel scrooling down 
				zoom *= 0.9f;
			}
		}

		void mouse_move(int x, int y) {
			if(dragging) {
				camera_angle_v += (y - drag_y_origin)*0.3;
				camera_angle_h += (x - drag_x_origin)*0.3;
				drag_x_origin = x;
				drag_y_origin = y;
			}
		}

		void reshape( int w, int h )
		{
			glViewport( 0, 0, (GLsizei) w, (GLsizei) h );
			glMatrixMode( GL_PROJECTION );
			glLoadIdentity( );
			glOrtho(0.0, (GLdouble)w, 0.0, (GLdouble)h, -1.0f, 100.0f);
		}

		void key_press( unsigned char key, int x, int y ){
			switch ( key )
			{
			case 27: // Escape key
				glutDestroyWindow ( window_id );
				break;
			}
		}

		void init( void (* renderFunc)( void ) ) 
		{
			int argc = 1;
			char *argv[1] = {(char*)"Something"};
			glutInit(&argc, argv);
			glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
			glutInitWindowSize( 800, 800 );
			glutInitWindowPosition( 100, 100 );
			window_id = glutCreateWindow( "Visualization of Vesselness" );

			// display
			glutDisplayFunc( renderFunc );
			glutIdleFunc( renderFunc );
			// register mouse fucntions
			glutMouseFunc( mouse_click );
			glutMotionFunc( mouse_move );
			// register keyboard functions
			glutKeyboardFunc( key_press );
			// window resize 
			glutReshapeFunc( reshape );

			glutMainLoop();
		}

		void show_dir( const Edge_Graph<Edge_Ext>& graph, const std::vector<Line>& lines ){
			ptrGraph = &graph;
			ptrLines = &lines;
			init( renderDirFunc );
		}
}

