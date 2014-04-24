#pragma once
#include "line3d.h"
class Line3DTwoPoint :
	public Line3D
{
	Vec6d data; 
public:
	Line3DTwoPoint(void);
	virtual ~Line3DTwoPoint(void);

	// Give a point, calculat the distance from the point to the line 
	virtual double distanceToLine( const Vec3d& point ) const; 

	// Given a point calculate the negative log likihood of a point being asign to this line
	virtual double loglikelihood( const Vec3d& point ) const; 
	virtual inline int getNumOfParameters( void ) { return 6; }

	// Given a point, return the projection point on the line
	virtual Vec3d projection( const Vec3d& point ) const;
	
	virtual inline void setPositions( const Vec3d& pos1, const Vec3d& pos2 );

	virtual void updateParameterWithDelta( int i, double delta ); 
	
	virtual Vec3d getDirection( void ) const;

	virtual void getEndPoints( Vec3d& p1, Vec3d& p2 ) const; 
};

