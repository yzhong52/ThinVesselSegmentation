#pragma once
#include "stdafx.h" 
class Line3D
{
public:
	Line3D(void);
	~Line3D(void);

	cv::Vec3f pos;
	// TODO: set setter for dir and provent dir without being normalized
	cv::Vec3f dir; 
	float sigma;
	
	// distance from a point to the line
	float distanceToLine( const Vec3f& point ) const;
	// projecting a point to the line
	Vec3f projection( const Vec3f& point ) const;
};

