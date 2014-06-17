#ifndef RINGSREDUCTION_H
#define RINGSREDUCTION_H

#include "Data3D.h"

class RingsReduction
{
public:
    static void unname_method( const Data3D<short>& src, Data3D<short>& dst );

    static void mm_filter( const Data3D<short>& src, Data3D<short>& dst );

private:
    // compute maximum number of rings
    static float max_ring_radius( const cv::Vec2f& center,
                                  const cv::Vec2f& im_size );

    // average intensity on ring rid
    // The actual radius of the ring will be rid * dr
    static double avgI_on_rings( const cv::Mat_<short>& m,
                                 const cv::Vec2f& ring_center,
                                 const int& rid,
                                 const double& dr );

    static void correct_image( const Data3D<short>& src, Data3D<short>& dst,
                               const std::vector<double>& correction,
                               const int& slice,
                               const cv::Vec2i& ring_center );
};

typedef RingsReduction RR;


#endif // RINGSREDUCTION_H
