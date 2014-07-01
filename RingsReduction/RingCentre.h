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
            const double& sigma = 1.0,
            const float& threshold_distance = 20.0f, // Threshold on the distance to the approximate centre
            const float& threshold_gradient = 2.4e3 ); // Threshold over the gradient

    /// compute the center of the ring with canny edge detector
    static cv::Vec2f canny_edges_method( const Data3D<short>& src,
                                         const cv::Vec2i& approx_center,
                                         const double& threshold1 = 0.0,
                                         const double& threshold2 = 0.0,
                                         const double& sigma = 1.89,
                                         const float& threshold_distance = 10.0f );

    /// Turn on Debug mode to save intermediate result for debugging
    static bool DEBUG_MODE;
    static std::string output_prefix;
    static void save_image( const std::string& name, const cv::Mat& im );

private:

    /// Get Centre through least square
    static cv::Vec2f least_square( const cv::Mat_<float>& grad_x,
                            const cv::Mat_<float>& grad_y,
                            const cv::Mat& mask );

    // computing distance from a point ('approx_centre') to a line ('point'
    // and 'dir')
    static float distance_to_line( const cv::Vec2i& approx_centre,
                                   const cv::Vec2i& point,
                                   const cv::Vec2f& dir );

    // Generate a mask from a distance to the centre
    static void threshold_distance_to_centre( const cv::Vec2i& approx_centre,
            const cv::Mat_<float>& grad_x,
            const cv::Mat_<float>& grad_y,
            const float& threshold,
            cv::Mat& mask );

    /// Computing the gradient of the iamge
    static void get_image_gradient( const cv::Mat_<short>& m,
                                    cv::Mat_<float>& grad_x,  // OUTPUT
                                    cv::Mat_<float>& grad_y,  // OUTPUT
                                    cv::Mat&         grad,    // OUTPUT
                                    const double& sigma );
};

#endif // RINGCENTRE_H
