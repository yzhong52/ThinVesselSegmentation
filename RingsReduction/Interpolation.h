#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <opencv2/core/core.hpp>
#include "smart_assert.h"

class Interpolation
{
public:

    /* Bilinear Interpolation */
    template<class T>
    static double bilinear( const cv::Mat_<T>& m,
                               const double& x, const double& y );

    /* ***Interpolation by Sampling***
      We assume that the shape of the pixel in polar coordinates corresponds to
      the following shape under the original Cartesian coordinates. The following
      shape is a rough approximation. The upper border and the lower border of the
      actual shape are both arcs instead of straight lines. And the actual shape
      may be of any orientations.
      <~~~~~~~~~~>
       \        /
        \      /
         ++-+-+                   */
    template<class T>
    static double Sampling( const cv::Mat_<T>& m,
                            const double& x, const double& y,
                            const cv::Vec2f& center,
                            const double& dangle, const double& dr );

private:

    /// transform from Cartesian coordinate to polar coordinate
    static void Cartecian2Polar( const cv::Vec2d& pos,
                                 const cv::Vec2f& center,
                                 double& radius,
                                 double& angle );

    /// Determine whether a given point is within the sector defined by 'center',
    /// 'dangle' and 'dr'.
    static bool InSector( const double& x, const double& y,
                          const cv::Vec2f& center,
                          const double& angle, const double& radius,
                          const double& dangle, const double& dradius );
};



template<class T>
double Interpolation::bilinear( const cv::Mat_<T>& m,
                                    const double& x, const double& y )
{
    const int fx = (int) floor( x );
    const int cx = (int) ceil( x );
    const int fy = (int) floor( y );
    const int cy = (int) ceil( y );

    smart_assert( fx>=0 && cx<m.cols && fy>=0 && cy<m.rows,
                  "Invalid input image position. Please call the following " <<
                  "function before computing interpolation. " << std::endl <<
                  "\t bool isvalid( cv::Mat_<short>&, double, double ); " );

    if( fx==cx && fy==cy )
    {
        return m(fy, fx);
    }
    else if( fx==cx )
    {
        return m(fy, fx) * (cy - y) +
               m(cy, fx) * (y - fy);
    }
    else if ( fy==cy )
    {
        return m(fy, fx) * (cx - x) +
               m(fy, cx) * (x - fx);
    }
    else
    {
        return m(fy, fx) * (cx - x) * (cy - y) +
               m(cy, fx) * (cx - x) * (y - fy) +
               m(fy, cx) * (x - fx) * (cy - y) +
               m(cy, cx) * (x - fx) * (y - fy);
    }
}

template<class T>
double Interpolation::Sampling( const cv::Mat_<T>& m,
                                const double& x, const double& y,
                                const cv::Vec2f& center,
                                const double& dangle, const double& dradius )
{
    double angle, radius;
    Interpolation::Cartecian2Polar( cv::Vec2d(x,y), center, radius, angle );


    const double temp_radius[2] =
    {
        radius - dradius, radius + dradius
    };
    const double temp_angle[2] =
    {
        angle - dangle, angle + dangle
    };

    cv::Vec2d corners[4];
    for( unsigned index=0; index<4; index++ )
    {
        const unsigned i = index % 2;
        const unsigned j = index / 2;
        corners[index][0] = temp_radius[i] * cos( temp_angle[j] );
        corners[index][1] = temp_radius[i] * sin( temp_angle[j] );
    }

    // Bounding Box
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::min();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::min();
    for( unsigned index=0; index<4; index++ )
    {
        minX = std::min( minX, std::floor( corners[index][0] ) );
        maxX = std::max( maxX, std::ceil ( corners[index][0] ) );
        minY = std::min( minY, std::floor( corners[index][1] ) );
        maxY = std::max( maxY, std::ceil ( corners[index][1] ) );
    }

    /// sub-pixel interpolation
    const double sub_pixel = 0.1; // TODO: can make this a parameter
    double sum = 0.0;
    int count = 0;
    for( double x=minX; x<=maxX; x+=sub_pixel )
    {
        for( double y=minY; y<=maxY; y+=sub_pixel )
        {
            double temp_angle, temp_radius;
            if( InSector(x, y, center, angle, radius, dangle, dradius ) ) {
                sum += m( (int)y, (int)x );
                count++;
            }
        }
    }

    return count ? sum / count : 0;
}



#endif // INTERPOLATION_H
