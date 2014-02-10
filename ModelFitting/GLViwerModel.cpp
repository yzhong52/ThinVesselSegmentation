#include "GLViwerModel.h"
#include "Line3D.h" 

using namespace GLViewer;

GLViwerModel::GLViwerModel(void)
{
}


GLViwerModel::~GLViwerModel(void)
{
}

void GLViwerModel::addModel( GCoptimization* ptrGC, vector<Line3D> lines, Vec3i size ){
	GLLineModel* lineModel = new GLLineModel( size );
	for( int z=0;z<size[2];z++ ) for( int y=0;y<size[1];y++ ) for( int x=0;x<size[0];x++ ){
		int lid = ptrGC->whatLabel( x + y*size[0] + z*size[0]*size[1] );
		Vec3f projection_point(0,0,0);
		if( lid!=lines.size() ) {
			const Vec3f& start = lines[lid].pos; 
			const Vec3f& dir = lines[lid].dir; 
			float t = ( Vec3f(1.0f*x,1.0f*y,1.0f*z)-start ).dot( dir );
			projection_point = start + dir * t; 
		}
		lineModel->data.at(x,y,z) = projection_point;
	}
	objs.push_back( lineModel );
}


void GLViwerModel::addModel( GLLineModel* lineModel ){
	objs.push_back( lineModel );
}
