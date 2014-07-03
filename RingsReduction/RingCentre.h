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
    static cv::Vec2f method_threshold_gradient( const Data3D<short>& src,
            const cv::Vec2i& approx_centre,
            const double& sigma = 1.0,
            const float& threshold_distance = 20.0f, // Threshold on the distance to the approximate centre
            const float& threshold_gradient = 9.7e2 ); // Threshold over the gradient

    /// compute the center of the ring with threshold gradient method
    static cv::Vec2f method_weighted_gradient( const Data3D<short>& src,
            const cv::Vec2i& approx_centre,
            const double& sigma = 1.0,
            const float& threshold_distance = 20.0f ); // Threshold over the gradient

    /// compute the center of the ring with canny edge detector
    static cv::Vec2f method_canny_edges( const Data3D<short>& src,
                                         const cv::Vec2i& approx_center,
                                         const double& threshold1 = 0.0,
                                         const double& threshold2 = 0.0,
                                         const double& sigma = 1.89,
                                         const float& threshold_distance = 10.0f );

    /// compute the center of the ring with canny edge detector
    static cv::Vec2f method_canny_edges_angle( const Data3D<short>& src,
            const cv::Vec2i& approx_center,
            const double& threshold1 = 0.0,
            const double& threshold2 = 0.0,
            const double& sigma = 1.89,
            const float& threshold_degree = 5.0f );

    /// Turn on Debug mode to save intermediate result for debugging
    static bool DEBUG_MODE;
    static std::string output_prefix;
    static void save_image( const std::string& name, const cv::Mat& im );

private:

    /// Get Centre through least square
    static cv::Vec2f least_square( const cv::Mat_<float>& grad_x,
                                   const cv::Mat_<float>& grad_y,
                                   const cv::Mat& mask );

    static cv::Vec2f weighted_least_square( const cv::Mat_<float>& grad_x,
                                            const cv::Mat_<float>& grad_y,
                                            const cv::Mat& weights );

    // computing distance from a point ('approx_centre') to a line ('point'
    // and 'dir')
    static float distance_to_line( const cv::Vec2i& approx_centre,
                                   const cv::Vec2i& point,
                                   const cv::Vec2f& dir );

    /// Generate a mask for the points with a gradient pointing to the centre of rings
    static void threshold_distance_to_centre( const cv::Vec2i& approx_centre,
            const cv::Mat_<float>& grad_x,
            const cv::Mat_<float>& grad_y,
            const float& threshold,
            cv::Mat& mask );

    /// Generate a mask for the points with a small angle to the centre of rings
    static void threshold_angle_to_centre( const cv::Vec2i& approx_centre,
                                           const cv::Mat_<float>& grad_x,
                                           const cv::Mat_<float>& grad_y,
                                           const float& threshold_degree,
                                           cv::Mat& mask );

    static void canny_edge( const cv::Mat_<short>& m,
                            const double& threshold1,
                            const double& threshold2,
                            const double& sigma,
                            cv::Mat& mask_edges  );

    /// Computing the gradient of the iamge
    static void get_image_gradient( const cv::Mat_<short>& m,
                                    cv::Mat_<float>& grad_x,  // OUTPUT
                                    cv::Mat_<float>& grad_y,  // OUTPUT
                                    cv::Mat&         grad,    // OUTPUT
                                    const double& sigma );
};

#endif // RINGCENTRE_H
