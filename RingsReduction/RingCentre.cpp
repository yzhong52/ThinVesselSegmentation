#include "RingCentre.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp> // For Canny, Gaussianblur and etc.

using namespace cv;

bool RingCentre::DEBUG_MODE = false;
const std::string RingCentre::output_prefix = "output/RingCentre - ";


cv::Vec2f RingCentre::threshold_gradient_method( const Data3D<short>& src,
        const Vec2i& approx_centre,
        const double& sigma,
        const float& threshold_distance,
        const float& threshold_gradient )
{
    // a slice of the data
    const cv::Mat_<short> m = src.getMat( src.SZ()/2 );

    // compute the derivatives (gradient) of the image
    Mat_<float> grad_x, grad_y;
    RingCentre::get_image_gradient( m, grad_x, grad_y, sigma );



    cv::Mat mask_theshold_gradient( m.rows, m.cols, CV_8U, Scalar(0) );


//    for( int y=0; y<m.rows; y++ )
//    {
//        for( int x=0; x<m.cols; x++ )
//        {
//            const float& dx = grad_x.at<float>(y,x);
//            const float& dy = grad_y.at<float>(y,x);
//
//            // skip if there is no gradient
//            if( sqrt( dx*dx+dy*dy )>grad_threshold )
//            {
//                mask_theshold_gradient.at<char>(y,x) = (char) 255;
//                if( point_line_distance( approx_center, Vec2i(x0, y0), Vec2f(dx, dy) ) < dist_threshold )
//                {
//                    mask.at<char>(y,x) = (char) 255;
//                }
//            }
//        }
//    }

//    cv::imshow( "mask", mask );
//    cv::imshow( "mask_theshold_gradient", mask_theshold_gradient );
//    cv::imwrite( "mask.png", mask );
//    cv::imwrite( "mask_theshold_gradient.png", mask_theshold_gradient );

    // TODO: the mask is not necessary, only for validation
    cv::Mat_<unsigned char> mask( m.rows, m.cols, (unsigned char)0 );
    threshold_distance_to_centre( approx_centre, grad_x, grad_y,
                                 threshold_distance, mask );
    if( DEBUG_MODE ) save_image("Mask - Distance to Centre", mask );

    return Vec2f(0,0);













    // visulize some arbitrary point on the mask and the gradient direction
//    cv::Mat grad_dir_sample( m.rows, m.cols, CV_8UC3, Scalar(0) );
//    cvtColor( mask, grad_dir_sample, CV_GRAY2BGR );
//
//    for( int i=0; i<500; i++ )
//    {
//        int x = rand() % m.cols;
//        int y = rand() %for( int y=0; y<m.rows; y++ )



//    {
//        for( int x=0; x<m.cols; x++ )
//        {
//            const float& dx = grad_x.at<float>(y,x);
//            const float& dy = grad_y.at<float>(y,x);
//            // skip if there is no gradient
//
//            if( point_line_distance( approx_center, Vec2i(x0, y0), Vec2f(dx, dy) ) < dist_threshold )
//            {
//                edges_mask.at<char>(y,x) = edges.at<char>(y,x);
//            }
//
//        }
//    }
//    m.rows;




//
//        const float& dx = grad_x.at<float>(y,x);
//        const float& dy = grad_y.at<float>(y,x);
//
//        if( mask.at<char>(y,x)!=0 )
//        {
//
//            Vec2f dir( dx, dy );
//            dir /= sqrt( dir.dot(dir) );
//            cv::line( grad_dir_sample,
//                      Point( x, y ),
//                      Point( (int)( (float)x + dir[0] * 10), (int)( (float)y + dir[1] * 10 ) ),
//                      Scalar( 255, 0, 255), /*thickness*/2, /*antialiased line*/8, 0 );
//        }
//    }
//    cv::imshow( "grad_dir", grad_dir_sample );






    // computing the centre by solvign the linear system A * X = B
    Mat_<float> A(0, 2);
    Mat_<float> B(0, 1);
    Mat_<float> X; // result (2, 1) matrix
    for( int y=0; y<m.rows; y++ )
    {
        for( int x=0; x<m.cols; x++ )
        {
            if( mask.at<char>(y,x) != (char) 255 ) continue;

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


// Generate a mask from a distance to the centre
void RingCentre::threshold_distance_to_centre( const cv::Vec2i& approx_centre,
        const cv::Mat_<float>& grad_x,
        const cv::Mat_<float>& grad_y,
        const float& threshold,
        cv::Mat_<unsigned char>& mask )
{
    smart_assert( grad_x.cols==grad_y.cols, "Size of the grad image should match. " );
    smart_assert( grad_x.rows==grad_y.rows, "Size of the grad image should match. " );

    const int& rows = grad_x.rows;
    const int& cols = grad_x.cols;
    for( int y=0; y<rows; y++ )
    {
        for( int x=0; x<cols; x++ )
        {
            const float& dx = grad_x.at<float>(y,x);
            const float& dy = grad_y.at<float>(y,x);
            // skip if there is no gradient

            if( point_line_distance( approx_centre, Vec2i(x,y), Vec2f(dx,dy) ) <  threshold )
            {
                mask.at<char>(y,x) = 255;
            }

        }
    }
}


cv::Vec2f RingCentre::canny_edges_method( const Data3D<short>& src,
        const cv::Vec2i& approx_center,
        const int& gksize,
        const double& threshold1,
        const double& threshold2 )
{

//    // Canny Edge Detector
//    cv::Mat cannyInput = m.clone();
//    cv::normalize( cannyInput, cannyInput, 0, 255, NORM_MINMAX, CV_8UC1 );
//    // cannyInput.convertTo( cannyInput, CV_8U );
//    cv::Mat edges;
//    cv::Canny( cannyInput, edges, 5800.0, 4500.0, 7, true );
//    cv::imshow( "Canny Edges", edges );
//    cv::imwrite( "Canny Edges.png", edges );
//
//    cv::Mat edges_mask( m.rows, m.cols, CV_8U, Scalar(0) );
//    for( int y=0; y<m.rows; y++ )
//    {
//        for( int x=0; x<m.cols; x++ )
//        {
//            const float& dx = grad_x.at<float>(y,x);
//            const float& dy = grad_y.at<float>(y,x);
//            // skip if there is no gradient
//
//            if( point_line_distance( approx_center, Vec2i(x0, y0), Vec2f(dx, dy) ) < dist_threshold )
//            {
//                edges_mask.at<char>(y,x) = edges.at<char>(y,x);
//            }
//
//        }
//    }
//    cv::imshow( "Canny Edges Threshold", edges_mask );
//    cv::imwrite( "Canny Edges Threshold.png", edges_mask );

    return Vec2f(0,0);
}



void RingCentre::get_image_gradient( const cv::Mat_<short>& m,
                                     Mat_<float>& grad_x,
                                     Mat_<float>& grad_y,
                                     const double& sigma )
{

    cv::Mat src_gray;
    m.convertTo( src_gray, CV_32F );
    cv::GaussianBlur( src_gray, src_gray,
                      cv::Size(0,0), /*Size is computed from sigma*/
                      sigma, sigma, /*sigma along x-y axis*/
                      BORDER_DEFAULT );

    int scale = 1;
    int delta = 0;
    int ddepth = CV_32F;

    /// Gradient X
    cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );

    /// Gradient Y
    cv::Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

    if( DEBUG_MODE )
    {
        // Saving Gradient X-Y
        save_image( "Gradient Y", grad_y );
        save_image( "Gradient X", grad_x );

        // Generate gradient x^2 + y^2
        Mat grad, grad_x2, grad_y2;
        cv::multiply( grad_x, grad_x, grad_x2 );
        cv::multiply( grad_y, grad_y, grad_y2 );
        cv::sqrt( grad_x2 + grad_y2, grad );
        save_image( "Gradient X2+Y2", grad );
    }
}



float RingCentre::point_line_distance( const cv::Vec2i& approx_centre,
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
    const float t = ( float(x-x0)*dx + float(y-y0)*dy ) / ( dx*dx + dy*dy );
    const float dist_x = x0 + t * dx - x;
    const float dist_y = y0 + t * dy - y;
    return sqrt( dist_x*dist_x + dist_y*dist_y );
}


void RingCentre::save_image( const string& name, const Mat& im )
{
    Mat dst;
    // normalize the image
    cv::normalize(im, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    // saving image
    cv::imwrite( output_prefix + name + ".png", dst );
}
