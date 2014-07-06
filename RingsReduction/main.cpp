#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "Data3D.h"
#include "GLViwerCore.h"

#include "RingCentre.h"
#include "RingsReduction.h"
#include "CVPlot.h"
#include "Interpolation.h"


using namespace std;
using namespace cv;

void save_slice( const Data3D<short>& im, const int& slice,
                 const double& minVal,
                 const double& maxVal,
                 const string& name )
{
    // save the centre slice of result
    Mat dst_norm = im.getMat( slice );
    // change the data type from whatever data type it is to float for computation
    dst_norm.convertTo(dst_norm, CV_32F);
    // normalize the data range from whatever it is to [0, 255];
    dst_norm = 255.0f * (  dst_norm - minVal ) / (maxVal - minVal);
    // convert back to CV_8U for visualization
    dst_norm.convertTo(dst_norm, CV_8U);
    // save file
    cv::imwrite( "output/" + name, dst_norm );
}

void save_slice( Mat im, const double& minVal, const double& maxVal,
                 const string& name, const Vec2i& center,
                 Vec3b center_color = Vec3b(0, 0, 255) )
{
    // change the data type from whatever data type it is to float for computation
    im.convertTo(im, CV_32F);
    // normalize the data range from whatever it is to [0, 255];
    im = 255.0f * (  im - minVal ) / (maxVal - minVal);
    // convert gray scale to color image
    cv::cvtColor(im, im, CV_GRAY2RGB);
    // convert back to CV_8U for visualization
    im.convertTo(im, CV_8UC3);
    // draw the centre point on the image
    im.at<Vec3b>(center[1], center[0]) = center_color;
    // save file
    cv::imwrite( "output/" + name, im );
}


#include <omp.h>
#include <stdio.h>
#include <stdlib.h>





int main(void)
{
    // loading data
    Data3D<short> im_short;
    bool flag = im_short.load( "../temp/vessel3d.data", Vec3i(585, 525, 10),
                               true, true );
    if( !flag ) return 0;


    if( false ) /*centre_detection*/
    {
        // calculating the centre of the ring
        RC::DEBUG_MODE = true;
        RC::output_prefix = "output/gradient/";
        Vec2f centre1 = RC::method_threshold_gradient( im_short, Vec2i( 234, 270 ) );
        cout << centre1 << endl;

        RC::output_prefix = "output/canny/";
        Vec2f centre2 = RC::method_canny_edges( im_short, cv::Vec2i( 234, 270 ) );
        cout << centre2 << endl;

        RC::output_prefix = "output/canny angle/";
        Vec2f centre3 = RC::method_canny_edges_angle( im_short, cv::Vec2i( 234, 270 ) );
        cout << centre3 << endl;

        RC::output_prefix = "output/weighted gradient/";
        Vec2f centre4 = RC::method_weighted_gradient( im_short, cv::Vec2i( 234, 270 ) );
        cout << centre4 << endl;

        cout << Vec2f(233.601f, 269.601f) << " - Expected Centre. "<< endl;


        waitKey(0);
        return 0;
    }

    // A approximation of the ring center
    const Vec2f apprx_centre( 233.8f, 269.8f );
    const Vec2f apprx_centre_i( 234, 270 );

    Interpolation<short>::Get = Interpolation<short>::Bilinear;
    Interpolation<short>::Get = Interpolation<short>::Sampling;

    if( true )   // distribution of intensity
    {
        RingsReduction::distri_of_diff( im_short.getMat( im_short.SZ()/2 ), apprx_centre, 100, 101, 1.0 );
        // return 0;
    }


    if( true ) /*ring_reduction*/
    {


        // compute the minimum value and maximum value in the centre slice
        double minVal = 0, maxVal = 0;
        cv::minMaxLoc( im_short.getMat( im_short.SZ()/2 ), &minVal, &maxVal );

        // save the original data with centre point
        Mat m = im_short.getMat( im_short.SZ()/2 );
        save_slice( m, minVal, maxVal, "original_234_270.png", Vec2i(234, 270) );
        save_slice( m, minVal, maxVal, "original_233_269.png", Vec2i(233, 269) );

        if( false ) // resize image
        {
            Mat big_im;
            cv::resize( m, big_im, m.size()*10 );
            save_slice( big_im, minVal, maxVal, "original big.png", Vec2i(2339, 2699) );
            return 0;
        }




        Data3D<short> im_rduct, im_rduct2;
        vector<double> correction;

        Interpolation<short>::Get = Interpolation<short>::Sampling;
        RR::AccumulatePolarRD( im_short, im_rduct, 1.0f, apprx_centre_i );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "minimize median difference - sampling.png" );

        Interpolation<short>::Get = Interpolation<short>::Bilinear;
        RR::AccumulatePolarRD( im_short, im_rduct, 1.0f, apprx_centre_i );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "minimize median difference - bilinear.png" );

return 0;

        RR::AccumulatePolarRD( im_short, im_rduct, 0.2f, apprx_centre );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_med_diff_v1_subpixel.png" );



        RR::polarRD( im_short, im_rduct, RR::AVG_DIFF, 1.0f );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_avg_diff.png" );

        RR::polarRD( im_short, im_rduct, RR::AVG_DIFF, 1.0f, apprx_centre  );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_avg_diff_subpixel.png" );

        RR::polarRD( im_short, im_rduct, RR::AVG_DIFF, 0.2f, apprx_centre, 0.2f );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_avg_diff_subpixel_0.2gap.png" );




        RR::polarRD( im_short, im_rduct, RR::MED_DIFF, 1.0f );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_med_diff.png" );

        RR::polarRD( im_short, im_rduct, RR::MED_DIFF, 1.0f, apprx_centre );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_med_diff_subpixel.png" );

        RR::polarRD( im_short, im_rduct, RR::MED_DIFF, 0.2f, apprx_centre, 0.2f  );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_med_diff_subpixel_0.2gap.png" );



        RR::sijbers( im_short, im_rduct, 1.0f );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "sijbers.png" );

        RR::sijbers( im_short, im_rduct, 1.0f, apprx_centre );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "sijbers_subpixel.png" );

        RR::sijbers( im_short, im_rduct, 0.2f, apprx_centre );
        save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "sijbers_subpixel_0.2dr.png" );
    }
    return 0;
}

