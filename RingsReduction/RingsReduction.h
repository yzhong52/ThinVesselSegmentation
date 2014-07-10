#ifndef RINGSREDUCTION_H
#define RINGSREDUCTION_H

#include "Data3D.h"
#include "Interpolation.h"

class RingsReduction;
typedef RingsReduction RR;

class RingsReduction
{
public:

    /// rings reduction in polar coordinates, different methods
    enum PolarRDOption
    {
        /// based on the difference between average intensities between rings
        AVG_DIFF,
        /// based on the difference between median intensities between rings
        MED_DIFF
    };
    static void polarRD( const Data3D<short>& src,  // Input Image
                         Data3D<short>& dst,        // Output Image
                         const PolarRDOption& o, const double dr = 1.0f,
                         const cv::Vec2d& approx_centre = cv::Vec2d(234, 270),
                         const double& subpixel_on_ring = 1.0f,
                         std::vector<double>* pCorrection = nullptr );

    /// An mutation of the above function (2D)
    /// computing the correction in a accumulative manner
    static void MMDPolarRD( const cv::Mat_<short>& src,
                            cv::Mat_<short>& dst,
                            const cv::Vec2d& centre = cv::Vec2d(234, 270),
                            const double dradius = 1.0f );

    /// An mutation of the above function
    /// computing the correction in a accumulative manner
    static void MMDPolarRD( const Data3D<short>& src, Data3D<short>& dst,
                            const cv::Vec2d& first_slice_centre,
                            const cv::Vec2d& last_slice_centre,
                            const double dradius = 1.0 );

    /// rings reduction using sijbers's methods
    static void sijbers( const Data3D<short>& src, Data3D<short>& dst,
                         const double& dr = 1.0f,
                         const cv::Vec2d& ring_centre = cv::Vec2d(234, 270),
                         bool isGaussianBlur = false,
                         std::vector<double>* pCorrection = nullptr );

    /// rings reduction using sijbers's methods (old implementation, deprecated)
    static void mm_filter( const Data3D<short>& src, Data3D<short>& dst );

private:

    /// compute maximum radius of the rings
    // distance from the centre of the ring to four corners of the image.
    static double max_ring_radius( const cv::Vec2d& center,
                                   const cv::Vec2d& im_size );

    /// Average intensity on ring rid (old implementation, deprecated)
    // The actual radius of the ring will be rid * dr
    static double avgI_on_rings( const cv::Mat_<short>& m,
                                 const cv::Vec2d& ring_center,
                                 const int& rid,
                                 const double& dr );

    /// Average difference between two rings
    static double avg_diff( const cv::Mat_<short>& m,
                            const cv::Vec2d& ring_center,
                            const int& rid1,
                            const int& rid2,
                            const double& dr );

    /// Median difference between two rings
    static double med_diff( const cv::Mat_<short>& m,
                            const cv::Vec2d& ring_center,
                            const int& rid1,
                            const int& rid2,
                            const double& dr );

    /// Average difference between two rings
    // This version (v2) is different from the one above that
    // it computes the average intensity of the rings separately and then
    // compute the difference. the above version compute them together.
    static double avg_diff_v2( const cv::Mat_<short>& m,
                               const cv::Vec2d& ring_center,
                               const int& rid1,
                               const int& rid2,
                               const double& dr,
                               const double& subpixel_on_ring = 1.0f );

    /// Median difference between two rings
    static double med_diff_v2( const cv::Mat_<short>& m,
                               const cv::Vec2d& ring_center,
                               const int& rid1,
                               const int& rid2,
                               const double& dr,
                               const double& subpixel_on_ring = 1.0f );

    /// average intensity on rings
    static double avg_on_ring( const cv::Mat_<short>& m,
                               const cv::Vec2d& ring_center,
                               const int& rid,
                               const double& dr,
                               const double& subpixel_on_ring = 1.0f );

    /// median intensity on ring
    template<class T>
    static double med_on_ring( const cv::Mat_<T>& m,
                               const cv::Vec2d& ring_center,
                               const int& rid,
                               const double& dr,
                               const double& subpixel_on_ring = 1.0f );

    /// Adjust image with give correction vector (2D)
    static void correct_image( const cv::Mat_<short>& src,
                               cv::Mat_<short>& dst,
                               const std::vector<double>& correction,
                               const cv::Vec2d& ring_center,
                               const double& dradius );

    /// Adjust image with give correction vector (3D)
    static void correct_image( const Data3D<short>& src, Data3D<short>& dst,
                               const std::vector<double>& correction,
                               const int& slice,
                               const cv::Vec2d& ring_center,
                               const double& dradius );
private:


    /// compute the median values in the vector
    // The order of the values in the vector in the following function
    static double median( std::vector<double>& values );


public:
    /// Utility functions for Yuri
    /* 1) Can I see the distribution of the difference of neighboring
        rings as histograms? Yes. */
    static std::vector<double> distri_of_diff( const cv::Mat_<short>& m,
            const cv::Vec2d& ring_center,
            const int& rid1, const int& rid2,
            const double& dr );
};





template<class T>
double RingsReduction::med_on_ring( const cv::Mat_<T>& m,
                                    const cv::Vec2d& ring_center,
                                    const int& rid,
                                    const double& dradius,
                                    const double& subpixel_on_ring )
{
    // radius of the circle
    const double radius = rid * dradius;

    // the number of pixels on the circumference approximately
    const int circumference = std::max( 8, int( 2 * M_PI * radius / subpixel_on_ring ) );

    std::vector<double> med(1, 0);

    const double dangle = 2 * M_PI / circumference;
    const double dangle_2 = dangle / 2;
    const double dradius_2 = dradius / 2;

    for( int i=0; i<circumference; i++ )
    {
        // angle in radian
        const double angle = i * dangle;
        const double sin_angle = sin( angle );
        const double cos_angle = cos( angle );

        // image position for inner circle
        const double x = radius * cos_angle + ring_center[0];
        const double y = radius * sin_angle + ring_center[1];

        if( Interpolation<T>::isvalid( m, x, y) )
        {
            const double val = Interpolation<T>::Get( m, cv::Vec2d(x,y), ring_center, dangle_2, dradius_2 );
            med.push_back( val );
        }
    }

    return median( med );
}








#endif // RINGSREDUCTION_H
