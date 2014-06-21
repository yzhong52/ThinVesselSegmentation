#ifndef RINGSREDUCTION_H
#define RINGSREDUCTION_H

#include "Data3D.h"

class RingsReduction
{
public:
    static void unname_method(   const Data3D<short>& src, Data3D<short>& dst );

    // rings reduction in polar coordinates, based on the difference between
    // average intensities between rings
    static void polar_avg_diff( const Data3D<short>& src, Data3D<short>& dst );

    static void mm_filter( const Data3D<short>& src, Data3D<short>& dst );

private:
    // compute maximum radius of the rings, distance from the centre of the ring
    // to four corners of the image.
    static float max_ring_radius( const cv::Vec2f& center,
                                  const cv::Vec2f& im_size );

    // average intensity on ring rid
    // The actual radius of the ring will be rid * dr
    static double avgI_on_rings( const cv::Mat_<short>& m,
                                 const cv::Vec2f& ring_center,
                                 const int& rid,
                                 const double& dr );
public:
    // average difference between two rings
    static double avg_diff( const cv::Mat_<short>& m,
                               const cv::Vec2f& ring_center,
                               const int& rid1,
                               const int& rid2,
                               const double& dr );
private:
    static double get( const cv::Mat_<short>& m, double x, double y );

    // give original image and a destination slice, correct_image (reduce rings)
    static void correct_image( const Data3D<short>& src, Data3D<short>& dst,
                               const std::vector<double>& correction,
                               const int& slice,
                               const cv::Vec2i& ring_center,
                               const double& dr );
};

typedef RingsReduction RR;


#endif // RINGSREDUCTION_H
