#pragma once

#include "Graph.h"
//#include "MinSpanTree.h"
#include "MinSpanTreeWrapper.h"
#include "DataTypes.h"

#include "GLViewer.h"
#include "gl\glew.h"
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library
#include "GL\freeglut.h"

namespace GLViewer{ 

	// abstract template class
	template<class T, class U=char>
	class CenterLine : public GLViewer::Object { };

	template<>
	class CenterLine<Edge> : public GLViewer::Object {
		MST::Graph3D<Edge>* ptrTree;
		// Original Data
		int sx, sy, sz;
		GLfloat red, green, blue;

		// rendeing mode
		enum RenderMode{ 
			POINTS,
			CONNECTIONS
		} render_mode; 
	public:
		CenterLine( MST::Graph3D<Edge>& tree ) : ptrTree( &tree ) { 
			sx = sy = sz = 0;
			for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
				Edge& e = ptrTree->get_edge(i);
				cv::Vec3i p1 = ptrTree->get_pos( e.node1 );
				cv::Vec3i p2 = ptrTree->get_pos( e.node2 );
				sx = (int) max( sx, p1[0] );
				sx = (int) max( sx, p2[0] );
				sy = (int) max( sy, p1[1] );
				sy = (int) max( sy, p2[1] );
				sz = (int) max( sz, p1[2] );
				sz = (int) max( sz, p2[2] );
			}
			// default color
			red = 0.0f; green = 1.0f; blue = 0.0f;
			// default rendering mode
			render_mode = CONNECTIONS; 
		}

		void init( void ) { }
		void setColor( GLfloat r, GLfloat g, GLfloat b ) {
			red = r; green = g; blue = b;
		}

		virtual void keyboard(unsigned char key ) {
			if ( key == '\t' ) { /*TAB key*/
				switch (render_mode)
				{
				case POINTS:
					render_mode = CONNECTIONS;
					break;
				case CONNECTIONS:
					render_mode = POINTS; 
					break; 
				}
			}
		}

		void render(void) {
			glColor3f( red, green, blue );
			if( !ptrTree->num_edges() ) return;

			if( render_mode == POINTS ){
				glBegin( GL_POINTS );
				for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
					Edge& e = ptrTree->get_edge(i);
					Vec3i p1 = ptrTree->get_pos( e.node1 );
					Vec3i p2 = ptrTree->get_pos( e.node2 );
					glVertex3i( p1[0], p1[1], p1[2] );
					glVertex3i( p2[0], p2[1], p2[2] );
				}
				glEnd();
			} else if( render_mode == CONNECTIONS ) {
				glBegin( GL_LINES );
				for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
					Edge& e = ptrTree->get_edge(i);
					Vec3i p1 = ptrTree->get_pos( e.node1 );
					Vec3i p2 = ptrTree->get_pos( e.node2 );
					glVertex3i( p1[0], p1[1], p1[2] );
					glVertex3i( p2[0], p2[1], p2[2] );
				}
				glEnd();
			} else {
				cout << "rendering mode is not defined" << endl;
				return;
			}
		}
		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }
	};

	// Centerline for Ryen's Line Fitting data
	template<>
	class CenterLine<MST::Edge_Ext, MST::LineSegment> : public GLViewer::Object {
		char mode; // display mode
		GLUquadric* quadric;
		MST::Graph< MST::Edge_Ext, MST::LineSegment >* ptrTree;
		int sx, sy, sz;
		GLfloat red, green, blue;
		GLfloat red2, green2, blue2;
	public:
		CenterLine( MST::Graph<MST::Edge_Ext, MST::LineSegment>& tree ) : ptrTree( &tree ) { 
			// get the size of the data
			sx = sy = sz = 0;
			for( unsigned int i=0; i<ptrTree->num_edges(); i++ ) {
				MST::Edge_Ext& e = ptrTree->get_edge(i);
				sx = (int) max( sx, e.line.p1.x );
				sx = (int) max( sx, e.line.p2.x );
				sy = (int) max( sy, e.line.p1.y );
				sy = (int) max( sy, e.line.p2.y );
				sz = (int) max( sz, e.line.p1.z );
				sz = (int) max( sz, e.line.p2.z );
			}
			for( unsigned int i=0; i< ptrTree->num_nodes(); i++ ) {
				MST::LineSegment& line = ptrTree->get_node( i );
				sx = (int) max( sx, line.p1.x );
				sx = (int) max( sx, line.p2.x );
				sy = (int) max( sy, line.p1.y );
				sy = (int) max( sy, line.p2.y );
				sz = (int) max( sz, line.p1.z );
				sz = (int) max( sz, line.p2.z );
			}
			// initialized the default color of the lines
			red =  1.0f; green = 0.0f;  blue = 0.0f;
			red2 = 0.0f; green2 = 1.0f; blue2 = 0.0f;
			// for rendering cylinder
			quadric = NULL;
			// decide what to display and what not to
			mode = 1;
		}
		CenterLine()
		{
			if( quadric ) {
				gluDeleteQuadric(quadric);
				quadric = NULL; 
			}
		}
		void setColor( GLfloat r, GLfloat g, GLfloat b, GLfloat r2, GLfloat g2, GLfloat b2 )
		{
			red = r;   green = g;   blue = b;
			red2 = r2; green2 = g2; blue2 = b2;
		}
		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }

		virtual void init(void) { 
			quadric = gluNewQuadric(); 
		}

		void drawCylinder( float x1, float y1, float z1, float x2, float y2, float z2, float radius ){
			// Yuchen: The following code is borrowed from a blog (sorry I cannot find it now)
			// But there is bug in the original code. I think I fix them. 

			float vx = x2-x1;
			float vy = y2-y1;
			float vz = z2-z1;

			float v = sqrt( vx*vx + vy*vy + vz*vz );
			
			static const double zero = 1.0e-3;
			static const double RADIAN_TO_DEGREE = 57.2957795; 

			glPushMatrix();
			glTranslatef( x1, y1, z1 );
			if( fabs(vz) < zero )
			{
				double ax = RADIAN_TO_DEGREE*acos( vx/v ); // rotation angle in x-y plane
				if ( vy >= 0.0 ) ax = -ax;    // Yuchen: fix bug
				glRotated(90.0, 0, 1, 0.0);   // Rotate & align with x axis
				glRotated(ax, 1.0, 0.0, 0.0); // Rotate to point 2 in x-y plane
			}
			else
			{
				// 1 radian is eqaul to 57.2957795 degrees
				double ax = RADIAN_TO_DEGREE*acos( vz/v ); // rotation angle
				if ( vz <= 0.0 ) ax = -ax;
				float rx = -vy*vz;
				float ry = vx*vz;
				glRotated(ax, rx, ry, 0.0); // Rotate about rotation vector
			}
			
			//draw the cylinder body
			gluQuadricOrientation(quadric,GLU_OUTSIDE);
			gluCylinder(quadric, radius, radius, v, 10, 1);

			//draw the first cap
			gluQuadricOrientation(quadric,GLU_INSIDE);
			gluDisk( quadric, 0.0, radius, 10, 1);
			glTranslatef( 0,0,v );

			//draw the second cap
			gluQuadricOrientation(quadric,GLU_OUTSIDE);
			gluDisk( quadric, 0.0, radius, 10, 1);
			glPopMatrix();
		}

		virtual void keyboard(unsigned char key ) {
			if ( key == '\t' ) { /*TAB key*/
				mode ++;
				// Round it over
				if( mode==0x7 ) mode = 0x1; 
			}
		}

		void render(void) {
			static char LINES = 0x1;
			static char CYLINDERS = 0x2;
			static char POINTS  = 0x4;
			
			if( mode & LINES ) {
				glBegin( GL_LINES );
				// draw 3d line
				glColor3f( red, green, blue );
				for( unsigned int i=0; i< ptrTree->num_nodes(); i++ ) {
					MST::LineSegment& line = ptrTree->get_node( i );
					glVertex3f( line.p1.x, line.p1.y, line.p1.z );
					glVertex3f( line.p2.x, line.p2.y, line.p2.z );
				}
				// draw connection, which are generated by min span tree
				glColor3f( red2, green2, blue2 );
				for( unsigned int i=0; i<ptrTree->num_edges(); i++ ) {
					MST::Edge_Ext& e = ptrTree->get_edge(i);
					glVertex3f( e.line.p1.x, e.line.p1.y, e.line.p1.z );
					glVertex3f( e.line.p2.x, e.line.p2.y, e.line.p2.z );
				}
				glEnd();
			}

			if(  mode & CYLINDERS ) {
				// draw 3d cylinders
				glColor3f( red, green, blue );
				for( unsigned int i=0; i< ptrTree->num_nodes(); i++ ) {
					MST::LineSegment& line = ptrTree->get_node( i );
					drawCylinder( line.p1.x, line.p1.y, line.p1.z, 
						line.p2.x, line.p2.y, line.p2.z, line.radius );

				}
				// draw connection, which are generated by min span tree
				glColor3f( red2, green2, blue2 );
				for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
					MST::Edge_Ext& e = ptrTree->get_edge(i);
					float radius = min( ptrTree->get_node(e.node1).radius, ptrTree->get_node(e.node2).radius );
					drawCylinder( e.line.p1.x, e.line.p1.y, e.line.p1.z, 
						e.line.p2.x, e.line.p2.y, e.line.p2.z, radius );
				}
			}

			if(  mode & POINTS ) {
				// draw point cloud
				glColor3f( 0.4f, 0.4f, 0.4f );
				glBegin( GL_POINTS );
				for( unsigned int i=0; i< ptrTree->num_nodes(); i++ ) {
					MST::LineSegment& line = ptrTree->get_node( i );
					vector<MST::Point3D>& points = line.points;
					for( unsigned int j=0; j<points.size(); j++ ) {
						glVertex3i( points[j].x, points[j].y, points[j].z );	
					}
				}
				glEnd();
			}
		}
	};
}