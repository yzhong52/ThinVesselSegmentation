#pragma once
#include "stdafx.h" 


// Line3D is a general abstract class
// For a line in 3D, there are many different representations: 
//     1) with a point and a direction (6 degrees of freedom)
//     2) with two points (6 degrees of freedom) 
//     3) with three eular engles and a distance to the origin (4 degrees of freedom) 
class Line3D
{
public:
	Line3D(void);
	virtual ~Line3D(void);

	// Give a point, calculat the distance from the point to the line 
	virtual float distanceToLine( const Vec3f& point ) const = 0;

	// Given a point calculate the negative log likihood of a point being asign to this line
	virtual float loglikelihood( const Vec3f& point ) const = 0; 

	virtual inline int getNumOfParameters( void ) = 0; 

	virtual void updateParameterWithDelta( int i, double delta ) = 0; 

	// Given a point, return the projection point on the line
	virtual Vec3f projection( const Vec3f& point ) const = 0;
};

