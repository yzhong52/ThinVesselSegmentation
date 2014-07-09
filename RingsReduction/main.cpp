#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "Data3D.h"
#include "GLViwerCore.h"

#include "RingCentre.h"
#include "RingsReduction.h"
#include "CVPlot.h"
#include "Interpolation.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

namespace experiment
{

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


void have_a_try(void)
{
    // loading data
    Data3D<short> im_short;
    bool flag = im_short.load( "../temp/vessel3d.data", Vec3i(585, 525, 10), true, true );
    if( !flag ) return;


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
        return;
    }

    // A approximation of the ring center
    const Vec2d apprx_centre_d( 233.7, 269.7 );
    const Vec2d apprx_centre_i( 234, 270 );

    const Vec2d apprx_centre_left(  233, 269 );
    const Vec2d apprx_centre_right( 234, 270 );

    if( true ) /*ring_reduction*/
    {
        // compute the minimum value and maximum value in the centre slice
        double minVal = 0, maxVal = 0;
        cv::minMaxLoc( im_short.getMat( im_short.SZ()/2 ), &minVal, &maxVal );

        if( false )  // save the original data with centre point
        {
            Mat m = im_short.getMat( im_short.SZ()/2 );
            save_slice( m, minVal, maxVal, "original_234_270.png", Vec2i(234, 270) );
            save_slice( m, minVal, maxVal, "original_233_269.png", Vec2i(233, 269) );
        }

        if( true )  // subsample experiment
        {
            Data3D<short> im_rduct;
            // vector<double> correction;

            const int subpixelcount = 10;
            for( int i=0; i<=subpixelcount; i++ )
            {
                cout << "i = " << i << endl;

                const Vec2d sub_centre = (i * apprx_centre_left + (subpixelcount-i) * apprx_centre_right) / subpixelcount;

                stringstream str1;
                str1 << "minimize median difference - sector - " << sub_centre << ".png";
                Interpolation<short>::Get = Interpolation<short>::Sampling;
                RR::AccumulatePolarRD( im_short, im_rduct, 1.0f, sub_centre );
                save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, str1.str() );

                stringstream str2;
                str2 << "minimize median difference - bilinear - " << sub_centre << ".png";
                Interpolation<short>::Get = Interpolation<short>::Bilinear;
                RR::AccumulatePolarRD( im_short, im_rduct, 1.0f, sub_centre );
                save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, str2.str() );

                stringstream str3;
                str3 << "Sijbers - Gaussian - bilinear - " << sub_centre << ".png";
                Interpolation<short>::Get = Interpolation<short>::Bilinear;
                Interpolation<float>::Get = Interpolation<float>::Bilinear;
                Interpolation<int>::Get   = Interpolation<int>::Bilinear;
                RR::sijbers( im_short, im_rduct, 1.0f, sub_centre, true );
                save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, str3.str() );

                stringstream str4;
                str4 << "Sijbers - Gaussian - sector - " << sub_centre << ".png";
                Interpolation<short>::Get = Interpolation<short>::Sampling;
                Interpolation<float>::Get = Interpolation<float>::Sampling;
                Interpolation<int>::Get   = Interpolation<int>::Sampling;
                RR::sijbers( im_short, im_rduct, 1.0f, sub_centre, true );
                save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, str4.str() );

            }

            return;
        }
    }
    return;
}

} // end of namespace



namespace state_of_the_art
{
void letsgo(void)
{
    // loading data
    Data3D<short> im_short;
    bool flag = im_short.load( "../temp/vessel3d.data" );
    if( !flag ) return;

    const Vec2d sub_centre( 233.5, 269.5 );

    Data3D<short> im_rduct;
    Interpolation<short>::Get = Interpolation<short>::Bilinear;
    Interpolation<float>::Get = Interpolation<float>::Bilinear;
    Interpolation<int>::Get   = Interpolation<int>::Bilinear;
    RR::sijbers( im_short, im_rduct, 1.0f, sub_centre, true );

    im_rduct.save( "../temp/vessel3d_rd.data" );
    im_rduct.show();
    return;
}
}


int main(void)
{
    const Vec2d apprx_centre_left(  233, 269 );
    const Vec2d apprx_centre_right( 234, 270 );

    const int subpixelcount = 10;
    for( int i=0; i<=subpixelcount; i++ )
    {
        cout << "i = " << i << endl;

        const Vec2d sub_centre = (i * apprx_centre_left + (subpixelcount-i) * apprx_centre_right) / subpixelcount;

        stringstream str1;
        str1 << "minimize median difference - sector - " << sub_centre << ".png";
        Mat mat;
        cv::imread(  );
        cv::show( "Mat::Image" );
        cv::waitKey( 0 );
    }

    // experiment::have_a_try();
    //state_of_the_art::letsgo();
    return 0;
}

