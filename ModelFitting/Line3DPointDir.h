#pragma once
#include "line3d.h"
class Line3DPointDir :
	public Line3D
{
public:
	Line3DPointDir(void);
	virtual ~Line3DPointDir(void);
public:
	// line data (vx, vy, vz, x0, y0, z0),
	// where (vx, vy, vz) is a normalized vector collinear to the line 
	// and (x0, y0, z0) is a point on the line
	cv::Vec3f pos; 
	cv::Vec3f dir; 
public:

	// Getters 
	inline const Vec3f& getPos(void) const { return pos; }
	inline const Vec3f& getDir(void) const { return dir; }
	// Setters
	inline void setPos( const Vec3f& position ) {  pos = position;  }
	inline void setDir( const Vec3f& direction ) { 
		dir = direction;
		float len2 = dir.dot( dir ); 
		// make sure the direction vector is normalized
		if( abs(len2)>1.0e-10 ) dir /= sqrt( len2 ); 
	}
	
	float sigma;
	
	// distance from a point to the line
	float distanceToLine( const Vec3f& point ) const;
	// projecting a point to the line
	Vec3f projection( const Vec3f& point ) const;
};

