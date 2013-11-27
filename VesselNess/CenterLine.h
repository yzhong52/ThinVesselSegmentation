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
		int sx;
		int sy;
		int sz;
	public:
		CenterLine( MST::Graph3D<Edge>& tree ) : ptrTree( &tree ) { 
			sx = 0;
			sy = 0;
			sz = 0;
		}

		void init( void ) { }

		void render(void) {
			if( !ptrTree->num_edges() ) return;
			glBegin( GL_LINES );
			Edge* e = &ptrTree->get_edges().top();
			glColor3f( 1.0f, 1.0f, 1.0f );
			for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
				Vec3i p1 = ptrTree->get_pos( e->node1 );
				Vec3i p2 = ptrTree->get_pos( e->node2 );
				glVertex3i( p1[0], p1[1], p1[2] );
				glVertex3i( p2[0], p2[1], p2[2] );
				e++;
			}
			glEnd();
			glColor3f( 0.9f, 0.1f, 0.1f );
		}
		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }
	};


	template<>
	class CenterLine<MST::Edge_Ext, MST::LineSegment> : public GLViewer::Object {
		Graph< MST::Edge_Ext, MST::LineSegment >* ptrTree;
		int sx, sy, sz;
	public:
		CenterLine( Graph<MST::Edge_Ext, MST::LineSegment>& tree ) : ptrTree( &tree ) { 
			sx = sy = sz = 0;
		}
		unsigned int size_x() const { return sx; }
		unsigned int size_y() const { return sy; }
		unsigned int size_z() const { return sz; }
		void render(void) {
			glBegin( GL_LINES );
			MST::Edge_Ext* e = &ptrTree->get_edges().top();
			glColor3f( 1.0f, 0.4f, 0.4f );
			for( int unsigned i=0; i<ptrTree->num_edges(); i++ ) {
				glVertex3f( e->line.p1.x, e->line.p1.y, e->line.p1.z );
				glVertex3f( e->line.p2.x, e->line.p2.y, e->line.p2.z );
				e++;
			}
			glColor3f( 0.4f, 0.4f, 1.0f );
			for( unsigned int i=0; i< ptrTree->num_nodes(); i++ ) {
				MST::LineSegment& line = ptrTree->get_node( i );
				glVertex3f( line.p1.x, line.p1.y, line.p1.z );
				glVertex3f( line.p2.x, line.p2.y, line.p2.z );
			}
			glEnd();
		}
	};
}