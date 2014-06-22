#ifndef RINGSREDUCTION_H
#define RINGSREDUCTION_H

#include "Data3D.h"

class RingsReduction
{
public:
    static void unname_method(   const Data3D<short>& src, Data3D<short>& dst );

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
                         std::vector<double>* pCorrection = nullptr );

    /// rings reduction using sijbers's methods
    static void sijbers( const Data3D<short>& src, Data3D<short>& dst,
                        std::vector<double>* pCorrection = nullptr );

    // rings reduction using sijbers's methods (olde implementation)
    static void mm_filter( const Data3D<short>& src, Data3D<short>& dst );

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

private:
    /// get the interpolation of the image data
    static double interpolate( const cv::Mat_<short>& m, double x, double y );

    /// Test if a image point (x,y) is valid or not
    static inline bool isvalid( const cv::Mat_<short>& m,
                                const double& x, const double& y )
    {
        return (x>=0 && x<=m.cols-1 && y>=0 && y<=m.rows-1);
    }

    /// Adjust image with give correction vector
    static void correct_image( const Data3D<short>& src, Data3D<short>& dst,
                               const std::vector<double>& correction,
                               const int& slice,
                               const cv::Vec2i& ring_center,
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

    std::vector<double> med;
    med.push_back( 0 );

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

#endif // RINGSREDUCTION_H
