#include "RingCentre.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // For Canny, Gaussianblur and etc.

using namespace cv;

bool RingCentre::DEBUG_MODE = false;
std::string RingCentre::output_prefix = "output/RingCentre - ";


cv::Vec2f RingCentre::method_threshold_gradient( const Data3D<short>& src,
        const Vec2i& approx_centre,
        const double& sigma,
        const float& threshold_distance,
        const float& threshold_gradient )
{
    // a slice of the data
    const cv::Mat_<short> m = src.getMat( src.SZ()/2 );

    // compute the derivatives (gradient) of the image
    Mat_<float> grad_x, grad_y;
    Mat grad;
    RingCentre::get_image_gradient( m, grad_x, grad_y, grad, sigma );

    if( DEBUG_MODE )
    {
        save_image( "Gradient Y", grad_y );
        save_image( "Gradient X", grad_x );
        save_image( "Gradient X2+Y2", grad );
    }

    cv::Mat mask;
    RingCentre::threshold_distance_to_centre( approx_centre, grad_x, grad_y,
            threshold_distance, mask );
    if( DEBUG_MODE )
    {
        save_image("Mask - Distance to Centre", mask );
    }

    cv::Mat mask2;
    cv::threshold( grad, mask2, threshold_gradient, 255, THRESH_BINARY );
    mask2.convertTo( mask2, CV_8U );

    if( DEBUG_MODE )
    {
        save_image("Mask - Gradient", mask2 );
    }


    Mat mask3;
    mask.copyTo( mask3, mask2 );
    if( DEBUG_MODE )
    {
        save_image("Mask - Distance to Centre && Gradient", mask3 );
    }

    return least_square( grad_x, grad_y, mask3 );
}


cv::Vec2f RingCentre::least_square( const cv::Mat_<float>& grad_x,
                                    const cv::Mat_<float>& grad_y,
                                    const cv::Mat& mask )
{
    smart_assert( grad_x.cols==grad_y.cols && grad_x.cols==mask.cols,
                  "Size of the grad image should match. " );
    smart_assert( grad_x.rows==grad_y.rows && grad_x.rows==mask.rows,
                  "Size of the grad image should match. " );

    smart_assert( mask.type()==CV_8U, "Mask must be of type <unsigned char>." )

    // size of the image
    const int& rows = grad_x.rows;
    const int& cols = grad_x.cols;

    /// computing the centre by solving the linear system A * X = B
    Mat_<float> A(0, 2);
    Mat_<float> B(0, 1);
    Mat_<float> X; // result (2, 1) matrix
    for( int y=0; y<rows; y++ )
    {
        for( int x=0; x<cols; x++ )
        {
            if( mask.at<unsigned char>(y, x) != (unsigned char)255 )
            {
                continue;
            }

            const float& dx = grad_x.at<float>(y,x);
            const float& dy = grad_y.at<float>(y,x);

            const float sqrt_dx2_dy2 = sqrt( dx*dx + dy*dy );

            Mat_<float> Arow(1, 2, 0.0f);
            Arow(0, 0) =  dy / sqrt_dx2_dy2;
            Arow(0, 1) = -dx / sqrt_dx2_dy2;
            A.push_back( Arow );

            Mat_<float> Brow(1, 1, 0.0f);
            Brow(0, 0) = float( (float)x * dy - (float)y * dx ) / sqrt_dx2_dy2;
            B.push_back( Brow );
        }
    }

    solve(A, B, X, DECOMP_SVD);

    return cv::Vec2f(X);
}



void RingCentre::threshold_angle_to_centre( const cv::Vec2i& approx_centre,
            const cv::Mat_<float>& grad_x,
            const cv::Mat_<float>& grad_y,
            const float& threshold_degree,
            cv::Mat& mask )
{
    smart_assert( grad_x.cols==grad_y.cols, "Size of the grad image should match. " );
    smart_assert( grad_x.rows==grad_y.rows, "Size of the grad image should match. " );

    const int& rows = grad_x.rows;
    const int& cols = grad_x.cols;

    mask = Mat( rows, cols, CV_8U, cv::Scalar(0) );

    const float threshold = std::abs( (float) std::cos( threshold_degree / 180 * M_PI + M_PI_2 ) );

    for( int y=0; y<rows; y++ )
    {
        for( int x=0; x<cols; x++ )
        {
            const float& dx = grad_x.at<float>(y,x);
            const float& dy = grad_y.at<float>(y,x);


            const Vec2f dir(-dy, dx);
            const Vec2f dir2( (float)(x - approx_centre[0]), (float)(y - approx_centre[1]) );

            const float dir_len = sqrt( dir.dot( dir ) );
            const float dir2_len = sqrt( dir2.dot( dir2) );

            if( dir_len < 1e-5 || dir2_len < 1e-5 ) continue;

            const float value = dir.dot( dir2 ) / ( dir_len * dir2_len );
            if( std::abs(value) < threshold )
            {
                mask.at<unsigned char>(y,x) = (unsigned char)255;
            }

        }
    }
}


void RingCentre::threshold_distance_to_centre( const cv::Vec2i& approx_centre,
        const cv::Mat_<float>& grad_x,
        const cv::Mat_<float>& grad_y,
        const float& threshold,
        cv::Mat& mask )
{
    smart_assert( grad_x.cols==grad_y.cols, "Size of the grad image should match. " );
    smart_assert( grad_x.rows==grad_y.rows, "Size of the grad image should match. " );

    const int& rows = grad_x.rows;
    const int& cols = grad_x.cols;

    mask = Mat( rows, cols, CV_8U, cv::Scalar(0) );

    for( int y=0; y<rows; y++ )
    {
        for( int x=0; x<cols; x++ )
        {
            const float& dx = grad_x.at<float>(y,x);
            const float& dy = grad_y.at<float>(y,x);

            if( distance_to_line( approx_centre, Vec2i(x,y), Vec2f(dx,dy) ) <  threshold )
            {
                mask.at<unsigned char>(y,x) = (unsigned char)255;
            }

        }
    }
}


void RingCentre::canny_edge( const cv::Mat_<short>& m,
                             const double& threshold1,
                             const double& threshold2,
                             const double& sigma,
                             cv::Mat& mask_edges )
{
    /// Input matrix for Canny Edge Detector
    cv::Mat cannyInput;
    m.convertTo( cannyInput, CV_32F );

    /// Blur the image
    cv::GaussianBlur( cannyInput, cannyInput,
                      cv::Size(0,0), /*Size is computed from sigma*/
                      sigma, sigma,  /*sigma along x and y axis*/
                      cv::BORDER_DEFAULT );

    /// Canny Edge Detector Input must be CV_8U
    cv::normalize( cannyInput, cannyInput, 0, 255, cv::NORM_MINMAX, CV_8UC1 );
    cannyInput.convertTo( cannyInput, CV_8U );

    const int apertureSize = 3;

    cv::Canny( cannyInput, mask_edges, threshold1, threshold2, apertureSize, true );
}

cv::Vec2f RingCentre::method_canny_edges( const Data3D<short>& src,
        const cv::Vec2i& approx_centre,
        const double& threshold1,
        const double& threshold2,
        const double& sigma,
        const float& threshold_distance )
{
    /// a slice of the data
    const cv::Mat_<short> m = src.getMat( src.SZ()/2 );

    /// Input matrix for Canny Edge Detector
    cv::Mat mask_edges;
    canny_edge( m, threshold1, threshold2, sigma, mask_edges );

    /// compute the derivatives (gradient) of the image
    Mat_<float> grad_x, grad_y;
    Mat grad;
    RingCentre::get_image_gradient( m, grad_x, grad_y, grad, sigma );

    /// Distance to the centre of the ring
    cv::Mat mask;
    RingCentre::threshold_distance_to_centre( approx_centre, grad_x, grad_y,
            threshold_distance, mask );

    Mat mask3;
    mask.copyTo( mask3, mask_edges );

    if( DEBUG_MODE )
    {
        save_image( "Mask - Canny Edges", mask_edges );
        save_image( "Gradient Y", grad_y );
        save_image( "Gradient X", grad_x );
        save_image( "Gradient X2+Y2", grad );
        save_image("Mask - Distance to Centre", mask );
        save_image("Mask - Distance to Centre && Canny", mask3 );
    }

    return least_square( grad_x, grad_y, mask3 );
}


cv::Vec2f RingCentre::method_canny_edges_angle( const Data3D<short>& src,
        const cv::Vec2i& approx_centre,
        const double& threshold1,
        const double& threshold2,
        const double& sigma,
        const float& threshold_degree )
{


        /// a slice of the data
    const cv::Mat_<short> m = src.getMat( src.SZ()/2 );

    /// Input matrix for Canny Edge Detector
    cv::Mat mask_edges;
    canny_edge( m, threshold1, threshold2, sigma, mask_edges );

    /// compute the derivatives (gradient) of the image
    Mat_<float> grad_x, grad_y;
    Mat grad;
    RingCentre::get_image_gradient( m, grad_x, grad_y, grad, sigma );

    /// Distance to the centre of the ring
    cv::Mat mask;
    RingCentre::threshold_angle_to_centre( approx_centre, grad_x, grad_y,
            threshold_degree, mask );

    Mat mask3;
    mask.copyTo( mask3, mask_edges );

    if( DEBUG_MODE )
    {
        save_image( "Mask - Canny Edges", mask_edges );
        save_image( "Gradient Y", grad_y );
        save_image( "Gradient X", grad_x );
        save_image( "Gradient X2+Y2", grad );
        save_image( "Mask - Angle", mask );
        save_image( "Mask - Angle && Canny", mask3 );
    }

    return least_square( grad_x, grad_y, mask3 );
}


void RingCentre::get_image_gradient( const cv::Mat_<short>& m,
                                     Mat_<float>& grad_x,
                                     Mat_<float>& grad_y,
                                     Mat& grad,
                                     const double& sigma )
{
    cv::Mat src_gray;
    m.convertTo( src_gray, CV_32F );
    cv::GaussianBlur( src_gray, src_gray,
                      cv::Size(0,0), /*Size is computed from sigma*/
                      sigma, sigma,  /*sigma along x-y axis*/
                      BORDER_DEFAULT );

    /// Some default settings
    const int scale = 1, delta = 0, ddepth = CV_32F;

    /// Gradient X
    cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

    /// Gradient Y
    cv::Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

    /// Generate gradient x^2 + y^2
    Mat grad_x2, grad_y2;
    cv::multiply( grad_x, grad_x, grad_x2 );
    cv::multiply( grad_y, grad_y, grad_y2 );
    cv::sqrt( grad_x2 + grad_y2, grad );
}



float RingCentre::distance_to_line( const cv::Vec2i& approx_centre,
                                    const cv::Vec2i& point,
                                    const cv::Vec2f& dir  )
{
    const int& x = approx_centre[0];
    const int& y = approx_centre[1];

    const int& x0 = point[0];
    const int& y0 = point[1];

    const float& dx = dir[0];
    const float& dy = dir[1];

    // an arbitrary point on the line (x0 + t * dx, y0 + t * dy )
    // distance from point (x, y) to a point on the line:
    //         (x0 + t * dx - x)^2 + (y0 + t * dy - y)^2
    // Derivative over t:
    //         2 * (x0 + t * dx - x) * dx + 2 * (y0 + t * dy - y) * dy = 0
    // Therefore t:
    //         t = ( (x-x0)*dx + (y-y0)*dy ) / ( dx*dx + dy*dy )
    const float t = ( float(x-x0)*dx + (float)(y-y0)*dy ) / ( dx*dx + dy*dy );
    const float dist_x = (float)x0 + t * dx - (float)x;
    const float dist_y = (float)y0 + t * dy - (float)y;
    return sqrt( dist_x*dist_x + dist_y*dist_y );
}


void RingCentre::save_image( const string& name, const Mat& im )
{
    Mat dst;
    // normalize the image
    cv::normalize(im, dst, 0, 255, NORM_MINMAX, CV_8UC1);

    // saving image
    const string filename = output_prefix + name + ".png";
    bool save_success = cv::imwrite( filename, dst );

    smart_assert( save_success, "Error whiling saving the file '" + filename + "'. "
                  "Maybe the destination folder does not exist? " );
}
