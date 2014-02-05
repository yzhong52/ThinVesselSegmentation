#pragma once

#include <windows.h>		// Header File For Windows
#include <queue>

#include "GLViewer.h"
#include <gl\gl.h>			// Header File For The OpenGL32 Library
#include <gl\glu.h>			// Header File For The GLu32 Library

#include "Vesselness.h"
#include "Data3D.h" 

namespace GLViewer
{
	// rendering object with Maximum Intenstiy Projection
	class Direction : public GLViewer::Object {
		/////////////////////////////////////////
		// Data
		///////////////////////
		Data3D<Vesselness_Sig> *ptrVnSig;
	public:
		Direction( Data3D<Vesselness_Sig>& vn_sig ) 
			: ptrVnSig( &vn_sig )
		{
		}

		~Direction() {
			ptrVnSig = NULL;
		}

		void init() {
			glDisable (GL_LINE_SMOOTH);
			// glHint (GL_LINE_SMOOTH_HINT, GL_NICEST );
		}
		void render(void){
			glBegin( GL_LINES );
			int gap = 2;
			for( int z=0; z<ptrVnSig->SZ(); z+=gap ) {
				for( int y=0; y<ptrVnSig->SY(); y+=gap ) {
					for( int x=0; x<ptrVnSig->SX(); x+=gap ) {
						if( ptrVnSig->at(x, y, z).rsp > 0.4f ) { 
							// select line color
							glColor4f( 1.0, 0.0, 0.0, ptrVnSig->at(x,y,z).rsp); 
							// draw line
							glVertex3i( x, y, z );
							glVertex3f( x + ptrVnSig->at(x,y,z).dir(0) * 5/* * ptrVnSig->at(x, y, z).rsp*/, 
								        y + ptrVnSig->at(x,y,z).dir(1) * 5/* * ptrVnSig->at(x, y, z).rsp*/, 
								        z + ptrVnSig->at(x,y,z).dir(2) * 5/* * ptrVnSig->at(x, y, z).rsp*/ );
						}
					}
				}
			}
			glEnd();
		}

		unsigned int size_x() const { return ptrVnSig->SX(); }
		unsigned int size_y() const { return ptrVnSig->SY(); }
		unsigned int size_z() const { return ptrVnSig->SZ(); }
	}; 
}