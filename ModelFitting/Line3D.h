#pragma once
#include "stdafx.h" 
#include <iostream>

// Line3D is a general abstract class
// For a line in 3D, there are many different representations: 
//     1) with a point and a direction (6 degrees of freedom)
//     2) with two points (6 degrees of freedom) 
//     3) with three eular engles and a distance to the origin (4 degrees of freedom) 
class Line3D
{
protected:
	// sigma describes the thickness of a line 
	// or can also be considered as the distribution factor
	double sigma;
public:
	// constructors 
	Line3D(void);
	virtual ~Line3D(void); 

	// setters 
	inline void setSigma( const double& s ){ sigma = s; }

	// Give a point, calculat the distance from the point to the line 
	virtual double distanceToLine( const Vec3d& point ) const = 0;

	// Given a point calculate the negative log likihood of a point being asign to this line
	virtual double loglikelihood( const Vec3d& point ) const = 0; 

	virtual int getNumOfParameters( void ) = 0; 

	virtual void updateParameterWithDelta( int i, double delta ) = 0; 

	// Given a point, return the projection point on the line
	virtual Vec3d projection( const Vec3d& point ) const = 0;

	virtual Vec3d getDirection( void ) const = 0; 

	virtual void getEndPoints( Vec3d& p1, Vec3d& p2 ) const = 0; 

	virtual void setPositions( const Vec3d& pos1, const Vec3d& pos2 ) = 0; 

	virtual void serialize( std::ostream& out ) const = 0;
	virtual void deserialize( std::istream& in ) = 0;
};

