#include "Interpolation.h"

using namespace cv;

void Interpolation::Cartecian2Polar( const cv::Vec2d& pos,
                                     const cv::Vec2f& center,
                                     double& radius,
                                     double& angle )
{
    /* A very good reference can be found [here]
       (http://en.wikipedia.org/wiki/Polar_coordinate_system)
       under the Section #Converting_between_polar_and_Cartesian_coordinates */

    const double rx = pos[0] - center[0];
    const double ry = pos[1] - center[1];

    /// Radius
    radius = std::sqrt( rx * rx + ry * ry );

    /// Angle: within range [ -0.5*PI, 1.5*PI ]
    if( std::abs(rx)>1e-10 )
    {
        angle = std::atan( ry / rx );
        if( rx < 0 )
        {
            if( ry >=0 ) angle += M_PI;
            else         angle -= M_PI;
        }
    }
    else if( std::abs(ry)>1e-10 )
    {
        angle = ( ry > 0 ) ? M_PI_2 : -M_PI_2;
    }
    else
    {
        smart_assert( 0, "Undefined conversion from Cartesian \
                     coordinates to polar coordinates. " );
    }
}


bool Interpolation::InSector( const double& x, const double& y,
                              const cv::Vec2f& center,
                              const double& angle, const double& radius,
                              const double& dangle, const double& dradius )
{
    /// Get the position under polar coordinates
    double this_angle, this_radius;
    Interpolation::Cartecian2Polar( Vec2d(x,y), center, this_radius, this_angle );

    if( std::abs( this_radius-radius) < dradius ) {
        if( std::abs(this_angle-angle) < dangle || std::abs(this_angle-angle-2*M_PI) < dangle ) {
            return true;
        }
    }
    return false;
}
