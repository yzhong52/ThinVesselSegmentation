#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <opencv2/core/core.hpp>
#include "smart_assert.h"

template<class T>
class Interpolation
{
public:

    // Function pointer: must be set before using
    static double (*Get)( const cv::Mat_<T>& m,
                                       const cv::Vec2d& pos,
                                       const cv::Vec2d& origin,
                                       const double& dangle,
                                       const double& dradius);


    /* ***Bilinear Interpolation***
    These parameters: three of the parameters are not used. */
    static double Bilinear( const cv::Mat_<T>& m,
                            const cv::Vec2d& pos,
                            const cv::Vec2d&,
                            const double&,
                            const double& );

    /* ***Interpolation by Sampling***
      We assume that the shape of the pixel in polar coordinates corresponds to
      the following shape under the original Cartesian coordinates. The following
      shape is a rough approximation. The upper border and the lower border of the
      actual shape are both arcs instead of straight lines. And the actual shape
      may be of any orientations.
       <~~~~~~~~~~>
        \        /
        \       /
         ++-+-+ */
    static double Sampling( const cv::Mat_<T>& m,
                            const cv::Vec2d& pos,
                            const cv::Vec2d& origin,
                            const double& dangle,
                            const double& dradius );

    /// Test if a image point (x,y) is valid or not
    static inline bool isvalid( const cv::Mat_<T>& m, const cv::Vec2d& pos );
    /// Test if a image point (x,y) is valid or not
    static inline bool isvalid( const cv::Mat_<T>& m,
                                const double& x, const double& y );
private:

    /// transform from Cartesian coordinate to polar coordinate
    static void Cartecian2Polar( const cv::Vec2d& pos,
                                 double& radius,
                                 double& angle );

    inline
    static void Cartecian2Polar( const cv::Vec2d& pos,
                                 const cv::Vec2d& origin,
                                 double& radius,
                                 double& angle );


    /// Determine whether a given point 'pos' is within the sector defined by
    /// 'origin', 'dangle' and 'dradius'.
    static bool InSector( const cv::Vec2d& pos,
                          const double& angle, const double& radius,
                          const double& dangle, const double& dradius );
};


template<class T>
double (*Interpolation<T>::Get)( const cv::Mat_<T>& m,
        const cv::Vec2d& pos,
        const cv::Vec2d& origin,
        const double& dangle,
        const double& dradius) = nullptr;


template<class T>
double Interpolation<T>::Bilinear( const cv::Mat_<T>& m, const cv::Vec2d& pos,
                                   const cv::Vec2d&, const double&, const double& )
{
    const double& x = pos[0];
    const double& y = pos[1];

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
double Interpolation<T>::Sampling( const cv::Mat_<T>& m,
                                   const cv::Vec2d& pos,
                                   const cv::Vec2d& center,
                                   const double& dangle,
                                   const double& dradius )
{
    double angle, radius;
    Interpolation<T>::Cartecian2Polar( pos, center, radius, angle );

    const double temp_radius[2] =
    {
        radius - dradius,
        radius + dradius
    };
    const double temp_angle[2] =
    {
        angle - dangle,
        angle + dangle
    };

    // Compute the four corners of the sector
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::min();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::min();
    for( unsigned index=0; index<4; index++ )
    {
        const unsigned i = index % 2;
        const unsigned j = index / 2;
        const double tempX = temp_radius[i] * cos( temp_angle[j] );
        const double tempY = temp_radius[i] * sin( temp_angle[j] );
        minX = std::min( minX, std::floor( tempX ) );
        maxX = std::max( maxX, std::ceil ( tempX ) );
        minY = std::min( minY, std::floor( tempY ) );
        maxY = std::max( maxY, std::ceil ( tempY ) );
    }

    /// sub-pixel interpolation
    const double sub_pixel = 0.2; // TODO: can make this a parameter

    double sum = 0.0;
    int count = 0;

    for( double y=minY; y<=maxY; y+=sub_pixel )
    {
        for( double x=minX; x<=maxX; x+=sub_pixel )
        {
            if( InSector( cv::Vec2f(x,y), angle, radius, dangle, dradius ) )
            {
                sum += m( (int)(y+center[1]), (int)(x+center[0]) );
                count++;
            }
        }
    }

    //double t1 = Interpolation<T>::Bilinear(m, pos, center, dangle, dradius);
    //double t2 = (count ? sum / count : 0);
    //std::cout << t1 << "\t" << t2 << "\t" << t1 - t2 << std::endl;

    return count ? sum / count : 0;
}


template<class T>
void Interpolation<T>::Cartecian2Polar( const cv::Vec2d& pos,
                                        const cv::Vec2d& origin,
                                        double& radius,
                                        double& angle )
{
    Interpolation<T>::Cartecian2Polar( pos-origin, radius, angle );
}

template<class T>
void Interpolation<T>::Cartecian2Polar( const cv::Vec2d& pos,
                                        double& radius,
                                        double& angle )
{
    /* A very good reference can be found [here]
       (http://en.wikipedia.org/wiki/Polar_coordinate_system)
       under the Section #Converting_between_polar_and_Cartesian_coordinates */

    const double& rx = pos[0];
    const double& ry = pos[1];

    /// Radius
    radius = std::sqrt( rx * rx + ry * ry );

    /// Angle: within range [ -0.5*PI, 1.5*PI ]
    if( std::abs(rx)>1e-10 )
    {
        angle = std::atan( ry / rx );
        if( rx < 0 )
        {
            if( ry >=0 ) angle += (double) M_PI;
            else         angle -= (double) M_PI;
        }
    }
    else if( std::abs(ry)>1e-10 )
    {
        angle = ( ry > 0 ) ? (double) M_PI_2 : (double)-M_PI_2;
    }
    else
    {
        // Undefined conversion from Cartesian coordinates to polar coordinates.
        angle = 0;
    }
}



template<class T>
bool Interpolation<T>::InSector( const cv::Vec2d& pos,
                                 const double& angle, const double& radius,
                                 const double& dangle, const double& dradius )
{
    /// Get the position under polar coordinates
    double this_angle, this_radius;
    Interpolation::Cartecian2Polar( pos, this_radius, this_angle );

    if( std::abs( this_radius-radius) < dradius )
    {
        if( std::abs(this_angle-angle) < dangle || std::abs(this_angle-angle-2*M_PI) < dangle )
        {
            return true;
        }
    }
    return false;
}


template<class T>
bool Interpolation<T>::isvalid( const cv::Mat_<T>& m, const cv::Vec2d& pos )
{
    return isvalid( m, pos[0], pos[1] );
}

template<class T>
bool Interpolation<T>::isvalid( const cv::Mat_<T>& m,
                                     const double& x, const double& y )
{
    return (x>=0 && x<=m.cols-1 && y>=0 && y<=m.rows-1);
}

#endif // INTERPOLATION_H
