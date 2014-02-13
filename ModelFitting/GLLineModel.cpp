#include "GLLineModel.h"

using namespace GLViewer;

void GLLineModel::render(void){
	glBegin( GL_POINTS );
	for( int z=0;z<data.SZ(); z++ )for( int y=0;y<data.SY();y++ )for( int x=0;x<data.SX();x++ ){
		const Vec3f& pos = data.at(x,y,z); 
		
		if( pos[0]>0 )
		{ 
			glColor3f(1.0f, 0.0f,0.0f); glVertex3f( pos[0], pos[1], pos[2] );
			// glColor3f(0.3f, 0.3f,0.3f); glVertex3f( 1.0f*x, 1.0f*y, 1.0f*z );
		}
	}
	glEnd();
}