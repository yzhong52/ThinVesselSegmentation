#include "Line3DTwoPoint.h"


Line3DTwoPoint::Line3DTwoPoint(void)
{
}


Line3DTwoPoint::~Line3DTwoPoint(void)
{
}


float Line3DTwoPoint::distanceToLine( const Vec3f& point ) const
{
	// get the porjection from the point to the line
	Vec3f proj_point = projection( point );
	Vec3f v = proj_point - point;
	return sqrt( v.dot(v) );
}

Vec3f Line3DTwoPoint::projection( const Vec3f& point ) const
{
	Vec3f& pos = Vec3f( &data[0] );
	Vec3f dir = Vec3f( &data[3] ) - Vec3f( &data[0] ); 
	dir /= sqrt( dir.dot( dir ) ); // normalize the direction 
	float t = ( point-pos ).dot( dir );
	return pos + dir * t; 
}


float Line3DTwoPoint::loglikelihood( const Vec3f& point ) const {
	float dist = this->distanceToLine( point ); 
	return dist * dist; // TODO: also should take care of sigma (devided by sigma here) 
}


void Line3DTwoPoint::updateParameterWithDelta( int i, double delta ) {
	data[i] += (float) delta; 
}