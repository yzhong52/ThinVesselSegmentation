#ifndef RINGSREDUCTION_H
#define RINGSREDUCTION_H

#include "Data3D.h"

class RingsReduction
{
public:
    static void unname_method( const Data3D<short>& src, Data3D<short>& dst );

    /// rings reduction in polar coordinates
    enum PolarRDOption
    {
        // based on the difference between average intensities between rings
        AVG_DIFF,
        // based on the difference between median intensities between rings
        MED_DIFF
    };
    static void polarRD( const Data3D<short>& src, Data3D<short>& dst,
                         const PolarRDOption& o, const float dr = 1.0f,
                         const float& center_x = 234,
                                   const float& center_y = 270,
                         std::vector<double>* pCorrection = nullptr );
    // an mutation of the above function
    // computing the correction in a accumulative manner
    static void AccumulatePolarRD( const Data3D<short>& src, Data3D<short>& dst,
                                   const PolarRDOption& o, const float dr = 1.0f,
                                   const float& center_x = 234,
                                   const float& center_y = 270,
                                   std::vector<double>* pCorrection = nullptr );

    /// rings reduction using sijbers's methods
    static void sijbers( const Data3D<short>& src, Data3D<short>& dst,
                        const float& dr = 1.0f,
                        const float& center_x = 234,
                        const float& center_y = 270,

                         std::vector<double>* pCorrection = nullptr );

    // rings reduction using sijbers's methods (olde implementation)
    static void mm_filter( const Data3D<short>& src, Data3D<short>& dst );

public:

    // Center of ring detection
    static cv::Vec2f get_ring_centre( const Data3D<short>& src,
                                      const cv::Vec2i& approx_center,
                                      const int& gksize = 11,
                                      const float& threshold = 10.0f );

public:

    static double dist( const int& x, const int& y,
                        const int& x0, const int& y0,
                        const float& dx, const float& dy );

    static void get_derivative( const cv::Mat_<short>& m,
                                cv::Mat_<float>& grad_x,
                                cv::Mat_<float>& grad_y,
                                const int& gksize  );

private:

    /// compute maximum radius of the rings
    // distance from the centre of the ring to four corners of the image.
    static float max_ring_radius( const cv::Vec2f& center,
                                  const cv::Vec2f& im_size );

    // average intensity on ring rid (old implementation)
    // The actual radius of the ring will be rid * dr
    static double avgI_on_rings( const cv::Mat_<short>& m,
                                 const cv::Vec2f& ring_center,
                                 const int& rid,
                                 const double& dr );

    /// Average difference between two rings
    static double avg_diff( const cv::Mat_<short>& m,
                            const cv::Vec2f& ring_center,
                            const int& rid1,
                            const int& rid2,
                            const double& dr );

    /// Median difference between two rings
    static double med_diff( const cv::Mat_<short>& m,
                            const cv::Vec2f& ring_center,
                            const int& rid1,
                            const int& rid2,
                            const double& dr );

    /// Average difference between two rings
    // This version (v2) is different from the one above that
    // it computes the average intensity of the rings seperately and then
    // compute the difference. the above version compute them togeter.
    static double avg_diff_v2( const cv::Mat_<short>& m,
                               const cv::Vec2f& ring_center,
                               const int& rid1,
                               const int& rid2,
                               const double& dr );

    /// Median difference between two rings
    static double med_diff_v2( const cv::Mat_<short>& m,
                               const cv::Vec2f& ring_center,
                               const int& rid1,
                               const int& rid2,
                               const double& dr );

    /// average intensity on rings
    static double avg_on_ring( const cv::Mat_<short>& m,
                               const cv::Vec2f& ring_center,
                               const int& rid,
                               const double& dr);

    /// median intensity on ring
    template<class T>
    static double med_on_ring( const cv::Mat_<T>& m,
                               const cv::Vec2f& ring_center,
                               const int& rid,
                               const double& dr);

    /// Adjust image with give correction vector
    static void correct_image( const Data3D<short>& src, Data3D<short>& dst,
                               const std::vector<double>& correction,
                               const int& slice,
                               const cv::Vec2i& ring_center,
                               const double& dr );
private:
    /// get the interpolation of the image data
    template<class T>
    static double interpolate( const cv::Mat_<T>& m,
                               const double& x, const double& y );

    /// Test if a image point (x,y) is valid or not
    template<class T>
    static inline bool isvalid( const cv::Mat_<T>& m,
                                const double& x, const double& y );

    /// compute the median values in the vector
    // the order of the values in the vector will be altered
    static double median( std::vector<double>& values );


public:
    /// Utility functions for Yuri
    // 1) Can I see the distribution of the difference of neiboring
    //    rings as histograms? Yes.
    static std::vector<double> distri_of_diff( const cv::Mat_<short>& m,
            const cv::Vec2f& ring_center,
            const int& rid1, const int& rid2,
            const double& dr );
};

typedef RingsReduction RR;



template<class T>
double RingsReduction::med_on_ring( const cv::Mat_<T>& m,
                                    const cv::Vec2f& ring_center,
                                    const int& rid,
                                    const double& dr)
{
    // radius of the circle
    const double radius = rid * dr;

    // the number of pixels on the circumference approximatly
    const int circumference = std::max( 8, int( 2 * M_PI * radius ) );

    std::vector<double> med(1, 0);

    for( int i=0; i<circumference; i++ )
    {
        // angle in radian
        const double angle = 2 * M_PI * i / circumference;
        const double sin_angle = sin( angle );
        const double cos_angle = cos( angle );

        // image possition for inner circle
        const double x = radius * cos_angle + ring_center[0];
        const double y = radius * sin_angle + ring_center[1];

        if( isvalid( m, x, y) )
        {
            const double val = interpolate( m, x, y );
            med.push_back( val );
        }
    }

    std::sort( med.begin(), med.end() );

    const double size = 0.5 * (double) med.size();
    const int id1 = (int) std::floor( size );
    const int id2 = (int) std::ceil( size );
    if( id1 == id2 )
    {
        return med[id1];
    }
    else
    {
        return 0.5 * ( med[id1] + med[id2] );
    }
}


template<class T>
double RingsReduction::interpolate( const cv::Mat_<T>& m,
                                    const double& x, const double& y )
{
    const int fx = (int) floor( x );
    const int cx = (int) ceil( x );
    const int fy = (int) floor( y );
    const int cy = (int) ceil( y );

    smart_assert( fx>=0 && cx<m.cols && fy>=0 && cy<m.rows,
                  "Invalid input image position. Please call the following " <<
                  "fucntion before computing interpolation. " << std::endl <<
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
inline bool RingsReduction::isvalid( const cv::Mat_<T>& m,
                                     const double& x, const double& y )
{
    return (x>=0 && x<=m.cols-1 && y>=0 && y<=m.rows-1);
}

#endif // RINGSREDUCTION_H
