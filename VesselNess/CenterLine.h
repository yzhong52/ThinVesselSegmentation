#pragma once

#include "Graph.h"
//#include "MinSpanTree.h"
#include "MinSpanTreeWrapper.h"
#include "DataTypes.h"

#include "GLViewer.h"
#include "gl\glew.h"
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library

namespace GLViewer{ 

	// abstract template class
	template<class T, class U=unsigned char>
	class CenterLine : public GLViewer::Object { };

	template<>
	class CenterLine<Edge> : public GLViewer::Object {
		MST::Graph3D<Edge>* ptrTree;
		// Original Data
		int sx, sy, sz;
		GLfloat red, green, blue;
	public:
		CenterLine( MST::Graph3D<Edge>& tree ) : ptrTree( &tree ) { 
			sx = sy = sz = 0;
			Edge* e = &ptrTree->get_edges().top();
			for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
				cv::Vec3i p1 = ptrTree->get_pos( e->node1 );
				cv::Vec3i p2 = ptrTree->get_pos( e->node2 );
				sx = (int) max( sx, p1[0] );
				sx = (int) max( sx, p2[0] );
				sy = (int) max( sy, p1[1] );
				sy = (int) max( sy, p2[1] );
				sz = (int) max( sz, p1[2] );
				sz = (int) max( sz, p2[2] );
				e++;
			}
			// default color
			red = 0.0f; green = 1.0f; blue = 0.0f;
		}

		void init( void ) { }
		void setColor( GLfloat r, GLfloat g, GLfloat b ) {
			red = r; green = g; blue = b;
		}
		void render(void) {
			glColor3f( red, green, blue );
			if( !ptrTree->num_edges() ) return;
			glBegin( GL_LINES );
			Edge* e = &ptrTree->get_edges().top();
			for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
				Vec3i p1 = ptrTree->get_pos( e->node1 );
				Vec3i p2 = ptrTree->get_pos( e->node2 );
				glVertex3i( p1[0], p1[1], p1[2] );
				glVertex3i( p2[0], p2[1], p2[2] );
				e++;
			}
			glEnd();
		}
		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }
	};


	template<>
	class CenterLine<MST::Edge_Ext, MST::LineSegment> : public GLViewer::Object {
		Graph< MST::Edge_Ext, MST::LineSegment >* ptrTree;
		int sx, sy, sz;
		GLfloat red, green, blue;
		GLfloat red2, green2, blue2;
	public:
		CenterLine( Graph<MST::Edge_Ext, MST::LineSegment>& tree ) : ptrTree( &tree ) { 
			sx = sy = sz = 0;
			MST::Edge_Ext* e = &ptrTree->get_edges().top();
			for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
				sx = (int) max( sx, e->line.p1.x );
				sx = (int) max( sx, e->line.p2.x );
				sy = (int) max( sy, e->line.p1.y );
				sy = (int) max( sy, e->line.p2.y );
				sz = (int) max( sz, e->line.p1.z );
				sz = (int) max( sz, e->line.p2.z );
				e++;
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
			red =  1.0f; green = 0.0f;  blue = 0.0f;
			red2 = 0.0f; green2 = 0.0f; blue2 = 1.0f;
		}
		void setColor( GLfloat r, GLfloat g, GLfloat b, GLfloat r2, GLfloat g2, GLfloat b2  ) {
			red = r;  green = g;  blue = b;
			red = r2; green = g2; blue = b2;
		}
		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }

		void render(void) {
			glBegin( GL_LINES );
			MST::Edge_Ext* e = &ptrTree->get_edges().top();
			glColor3f( red, green, blue );
			for( unsigned int i=0; i< ptrTree->num_nodes(); i++ ) {
				MST::LineSegment& line = ptrTree->get_node( i );
				glVertex3f( line.p1.x, line.p1.y, line.p1.z );
				glVertex3f( line.p2.x, line.p2.y, line.p2.z );
			}
			glColor3f( red2, green2, blue2 );
			for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
				glVertex3f( e->line.p1.x, e->line.p1.y, e->line.p1.z );
				glVertex3f( e->line.p2.x, e->line.p2.y, e->line.p2.z );
				e++;
			}
			glEnd();
		}
	};
}