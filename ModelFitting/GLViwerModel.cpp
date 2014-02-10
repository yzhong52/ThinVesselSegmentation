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
			projection_point = lines[lid].projection( Vec3f(1.0f*x,1.0f*y,1.0f*z) ); 
		}
		lineModel->data.at(x,y,z) = projection_point;
	}
	objs.push_back( lineModel );
}

void GLViwerModel::addModel(
	vector<Line3D>& lines, // the labels
	vector<vector<Vec3i> >& pointsSet, // there corresponding points
	cv::Vec3i& size ) // the size of the pointsSet
{
	smart_assert( lines.size()==pointsSet.size(), "Data Size don't match" );

	GLLineModel* lineModel = new GLLineModel( size );
	for( int i=0; i<lines.size(); i++ ){
		const Line3D& line = lines[i];
		for( int j=0; j<pointsSet[i].size(); j++ ) {
			const Vec3i point = pointsSet[i][j]; 
			Vec3f proj_point = line.projection( Vec3f(1.0f*point[0],1.0f*point[1],1.0f*point[2]) ); 
			lineModel->data.at( point ) = proj_point;
		}
	}
	objs.push_back( lineModel );
}


void GLViwerModel::addModel( GLLineModel* lineModel ){
	objs.push_back( lineModel );
}
