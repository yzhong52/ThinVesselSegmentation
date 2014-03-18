#include "Line3DPointDir.h"

 

Line3DPointDir::Line3DPointDir(void) 
	: sigma( 1.0f )
{
	setPos( Vec3f(0,0,0) ); 
	setDir( Vec3f(1,0,0) ); 
}


Line3DPointDir::~Line3DPointDir(void)
{
}


float Line3DPointDir::distanceToLine( const Vec3f& point ) const {
	// get the porjection from the point to the line
	Vec3f proj_point = projection( point );
	Vec3f v = proj_point - point;
	return sqrt( v.dot(v) );
}

Vec3f Line3DPointDir::projection( const Vec3f& point ) const{
	float t = ( point-pos ).dot( dir );
	return pos + dir * t; 
}