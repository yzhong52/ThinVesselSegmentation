#include <iostream>

#include "Data3D.h"
#include "GLViwerCore.h"
#include "RingsReduction.h"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void save_slice( const Data3D<short>& im, const int& slice,
                 const double& minVal,
                 const double& maxVal,
                 const string& name )
{
    // save the centre slice of result
    Mat dst_norm = im.getMat( slice );
    // change the data type from whatever dataype it is to float for computation
    dst_norm.convertTo(dst_norm, CV_32F);
    // normalize the data range from whatever it is to [0, 255];
    dst_norm = 255.0f * (  dst_norm - minVal ) / (maxVal - minVal);
    // convert back to CV_8U for visualizaiton
    dst_norm.convertTo(dst_norm, CV_8U);
    // save file
    cv::imwrite( "output/" + name, dst_norm );
}

int main()
{
    // laoding data
    Data3D<short> im_short;
    bool flag = im_short.load( "../temp/vessel3d.data", Vec3i(585, 525, 10), true, true );
    if( !flag ) return 0;

    // compute the minimum value and maximum value in the centre slice
    double minVal = 0, maxVal = 0;
    cv::minMaxLoc( im_short.getMat( im_short.SZ()/2 ), &minVal, &maxVal );

    // save the original data
    save_slice( im_short, im_short.SZ()/2, minVal, maxVal, "original.png" );

    Data3D<short> im_rduct;

    RR::polarRD( im_short, im_rduct, RR::AVG_DIFF, 1.0f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_avg_diff.png" );

    RR::polarRD( im_short, im_rduct, RR::MED_DIFF, 1.0f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_med_diff.png" );


    RR::sijbers( im_short, im_rduct );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "sijbers.png" );

    return 0;
}

