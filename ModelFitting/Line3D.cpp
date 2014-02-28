#include "Line3D.h"


Line3D::Line3D(void) 
	: sigma( 1.0f )
{
	setPos( Vec3f(0,0,0) ); 
	setDir( Vec3f(1,0,0) ); 
}


Line3D::~Line3D(void)
{
}


float Line3D::distanceToLine( const Vec3f& point ) const {
	// get the porjection from the point to the line
	Vec3f proj_point = projection( point );
	Vec3f v = proj_point - point;
	return sqrt( v.dot(v) );
}

Vec3f Line3D::projection( const Vec3f& point ) const{
	float t = ( point-pos ).dot( dir );
	return pos + dir * t; 
}