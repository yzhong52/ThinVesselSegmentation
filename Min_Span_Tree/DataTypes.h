#pragma once

#include "Edge.h"

/////////////////////////////////////////////////////////////
// Some basic data structures
/////////////////////////////////////////////////////////////
struct Vec3f{
	float x, y, z;
	// constructor
	Vec3f(const Vec3f&v = Vec3f(0,0,0) ) : x(v.x), y(v.y), z(v.z) { }
	Vec3f( float x, float y, float z ) : x(x), y(y), z(z) { }
	// operator
	const float dot( const Vec3f& v ) const { return x * v.x + y * v.y + z * v.z; } 
	inline Vec3f operator-( const Vec3f& v) const { return Vec3f( x-v.x, y-v.y, z-v.z); }
	inline Vec3f operator+( const Vec3f& v) const { return Vec3f( x+v.x, y+v.y, z+v.z); }
	inline Vec3f operator*( const float& v) const { return Vec3f( v*x, v*y, v*z );      }
};
struct LineSegment{
	Vec3f p1, p2;
	LineSegment( ) : p1( Vec3f(0,0,0) ), p2( Vec3f(0,0,0) ) { }
	LineSegment(const LineSegment& l ) : p1(l.p1), p2(l.p2) { }
	void get_distance( const Vec3f& from, 
		/*Output*/ Vec3f& to, 
		/*Output*/ float& distance )
	{
		Vec3f temp1 = p2 - p1;
		Vec3f temp2 = p1 - from;
		float numerator = temp2.dot( temp1 );
		float denominator = temp1.dot( temp1 );
		float t = -numerator / denominator;
		t = std::min( std::max(1.0f, t), 0.0f);
		to = p2 * t + p1 * (1-t);
		Vec3f dir = from - to;
		distance = dir.dot( dir );
	}
};
struct Edge_Ext : public Edge {
	LineSegment line;
};