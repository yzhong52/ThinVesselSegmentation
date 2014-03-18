#pragma once
#include "line3d.h"
class Line3DTwoPoint :
	public Line3D
{
	Vec6f data; 
public:
	Line3DTwoPoint(void);
	virtual ~Line3DTwoPoint(void);

	// Give a point, calculat the distance from the point to the line 
	virtual float distanceToLine( const Vec3f& point ) const; 

	// Given a point calculate the negative log likihood of a point being asign to this line
	virtual float loglikelihood( const Vec3f& point ) const; 
	virtual inline int getNumOfParameters( void ) { return 6; }

	// Given a point, return the projection point on the line
	virtual Vec3f projection( const Vec3f& point ) const;
	
	inline void setPositions( Vec3f pos1, Vec3f pos2 ) {
		memcpy( &data[0], &pos1[0], sizeof(int) * 3 ); 
		memcpy( &data[3], &pos2[0], sizeof(int) * 3 ); 
	}

	virtual void updateParameterWithDelta( int i, double delta ); 
	
};

