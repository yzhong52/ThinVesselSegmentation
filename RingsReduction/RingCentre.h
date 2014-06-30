#ifndef RINGCENTRE_H
#define RINGCENTRE_H

#include "Data3D.h"

class RingCentre;
typedef RingCentre RC;

// This class contains static functions for computing the center of ring
class RingCentre
{
public:

    /// compute the center of the ring with threshold gradient method
    static cv::Vec2f threshold_gradient_method( const Data3D<short>& src,
            const cv::Vec2i& approx_centre,
            const double& sigma,
            const float& threshold_distance = 10.0f, // Threshold on the distance to the approximate centre
            const float& threshold_gradient = 100 );      // Threshold over the gradient

    /// compute the center of the ring with canny edge detector
    static cv::Vec2f canny_edges_method( const Data3D<short>& src,
                                         const cv::Vec2i& approx_center,
                                         const int& gksize,
                                         const double& threshold1,
                                         const double& threshold2 );

    /// Turn on Debug mode to save intermediate result for debuging
    static bool DEBUG_MODE;
    static const std::string output_prefix;
    static void save_image( const std::string& name, const cv::Mat& im );

private:

    // computing distance from a point ('approx_centre') to a line ('point'
    // and 'dir')
    static float point_line_distance( const cv::Vec2i& approx_centre,
                                      const cv::Vec2i& point,
                                      const cv::Vec2f& dir );

    // Generate a mask from a distance to the centre
    static void threshold_distance_to_centre( const cv::Vec2i& approx_centre,
            const cv::Mat_<float>& grad_x,
            const cv::Mat_<float>& grad_y,
            const float& threshold,
            cv::Mat_<unsigned char>& mask );

    /// Computing the gradient of the iamge
    static void get_image_gradient( const cv::Mat_<short>& m,
                                    cv::Mat_<float>& grad_x,
                                    cv::Mat_<float>& grad_y,
                                    const double& sigma );
};

#endif // RINGCENTRE_H
