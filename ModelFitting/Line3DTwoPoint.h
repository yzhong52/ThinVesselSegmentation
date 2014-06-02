#pragma once
#include "Line3D.h"
class Line3DTwoPoint :
    public Line3D
{
    cv::Vec6d data;
public:
    Line3DTwoPoint(void);
    virtual ~Line3DTwoPoint(void);

    // Give a point, calculat the distance from the point to the line
    virtual double distanceToLine( const cv::Vec3d& point ) const;

    // Given a point calculate the negative log likihood of a point being asign to this line
    virtual double loglikelihood( const cv::Vec3d& point ) const;
    virtual inline int getNumOfParameters( void )
    {
        return 6;
    }

    // Given a point, return the projection point on the line
    virtual cv::Vec3d projection( const cv::Vec3d& point ) const;

    virtual inline void setPositions( const cv::Vec3d& pos1, const cv::Vec3d& pos2 );

    virtual void updateParameterWithDelta( int i, double delta );

    virtual cv::Vec3d getDirection( void ) const;

    virtual void getEndPoints( cv::Vec3d& p1, cv::Vec3d& p2 ) const;

    virtual void serialize( std::ostream& out ) const;
    virtual void deserialize( std::istream& in );
};


inline void Line3DTwoPoint::setPositions( const cv::Vec3d& pos1, const cv::Vec3d& pos2 )
{
    memcpy( &data[0], &pos1[0], sizeof(double) * 3 );
    memcpy( &data[3], &pos2[0], sizeof(double) * 3 );
}
