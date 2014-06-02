#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>

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

    // setters & getters
    inline void setSigma( const double& s )
    {
        sigma = s;
    }
    inline const double& getSigma(void) const
    {
        return sigma;
    }

    // Give a point, calculat the distance from the point to the line
    virtual double distanceToLine( const cv::Vec3d& point ) const = 0;

    // Given a point calculate the negative log likihood of a point being asign to this line
    virtual double loglikelihood( const cv::Vec3d& point ) const = 0;

    virtual int getNumOfParameters( void ) = 0;

    virtual void updateParameterWithDelta( int i, double delta ) = 0;

    // Given a point, return the projection point on the line
    virtual cv::Vec3d projection( const cv::Vec3d& point ) const = 0;

    virtual cv::Vec3d getDirection( void ) const = 0;

    virtual void getEndPoints( cv::Vec3d& p1, cv::Vec3d& p2 ) const = 0;

    virtual void setPositions( const cv::Vec3d& pos1, const cv::Vec3d& pos2 ) = 0;

    virtual void serialize( std::ostream& out ) const = 0;
    virtual void deserialize( std::istream& in ) = 0;
};

