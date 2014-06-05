#ifndef RINGSREDUCTION_H
#define RINGSREDUCTION_H

#include "Data3D.h"

class RingsReduction
{
public:
    static void a_dummy_method( const Data3D<short>& src, Data3D<short>& dst );

    static void mm_filter( const Data3D<short>& src, Data3D<short>& dst );

private:
    // compute maximum number of rings
    static int max_ring_radius( const cv::Vec2i& ring_center,
                                const cv::Vec2i& im_size );

    // average intensity on ring r
    static double avgI_on_rings( const cv::Mat_<short>& m,
                                 const cv::Vec2i& ring_center,
                                 const double& r,
                                 const double& dr = 1.0 );
};

typedef RingsReduction RR;


#endif // RINGSREDUCTION_H
