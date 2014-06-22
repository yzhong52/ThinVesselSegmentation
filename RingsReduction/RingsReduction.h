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
                         const PolarRDOption& o, const float dr = 1.0f );

    /// rings reduction using sijbers's methods
    static void sijbers( const Data3D<short>& src, Data3D<short>& dst );

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
    static double med_on_ring( const cv::Mat_<short>& m,
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


#endif // RINGSREDUCTION_H
