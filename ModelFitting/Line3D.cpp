#include "Line3D.h"


Line3D::Line3D(void) 
	: pos( Vec3f(0,0,0) )
	, dir( Vec3f(1,0,0) )
	, sigma( 1.0f )
{
}


Line3D::~Line3D(void)
{
}


float Line3D::distanceToLine( const Vec3f& point ) const {
	// get the porjection point
	Vec3f proj_point = projection( point );
	Vec3f v = proj_point - point;
	return sqrt( v.dot(v) );
}

Vec3f Line3D::projection( const Vec3f& point ) const{
	float t = ( point-pos ).dot( dir );
	return pos + dir * t; 
}