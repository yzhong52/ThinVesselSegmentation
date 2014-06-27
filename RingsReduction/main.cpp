#include <iostream>

#include "Data3D.h"
#include "GLViwerCore.h"
#include "RingsReduction.h"
#include "CVPlot.h"

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

void save_slice( Mat im, const double& minVal, const double& maxVal,
                 const string& name, const Vec2i& center,
                 Vec3b center_color = Vec3b(0, 0, 255) )
{
    // change the data type from whatever dataype it is to float for computation
    im.convertTo(im, CV_32F);
    // normalize the data range from whatever it is to [0, 255];
    im = 255.0f * (  im - minVal ) / (maxVal - minVal);
    // convert grayscale to color image
    cv::cvtColor(im, im, CV_GRAY2RGB);
    // convert back to CV_8U for visualizaiton
    im.convertTo(im, CV_8UC3);
    // draw the cnetre point on the image
    im.at<Vec3b>(center[1], center[0]) = center_color;
    // save file
    cv::imwrite( "output/" + name, im );
}


int main()
{
    // laoding data
    Data3D<short> im_short;
    bool flag = im_short.load( "../temp/vessel3d.data", Vec3i(585, 525, 10),
                               true, true );
    if( !flag ) return 0;

    // calculating the centre of the ring
    Vec2f centre = RR::get_ring_centre( im_short, cv::Vec2i( 234, 270 ), 31, 20 );
//    cout << centre << endl;
//    waitKey(0);
//    return 0;




    vector<double> diffs = RR::distri_of_diff( im_short.getMat( im_short.SZ()/2 ),
                           cv::Vec2f( 234, 270 ), 50, 51, 1.0f );
    CVPlot::draw( "distri_of_diff.png", diffs );

    const unsigned num_of_bins = 200;
    const unsigned num_of_diffs = (unsigned) diffs.size();
    std::sort( diffs.begin(), diffs.end() );
    vector<double> bins(num_of_bins, 0);
    double diff_range = diffs.back() - diffs.front();
    for( unsigned i=0; i<num_of_diffs; i++ )
    {
        unsigned binid = (unsigned) (num_of_bins * (diffs[i] - diffs.front()) / diff_range);
        binid = std::min( binid, num_of_bins-1);
        bins[binid]++;
    }

    CVPlot::draw( "distri_of_diff_sorted_bin.png", bins );



    // compute the minimum value and maximum value in the centre slice
    double minVal = 0, maxVal = 0;
    cv::minMaxLoc( im_short.getMat( im_short.SZ()/2 ), &minVal, &maxVal );

    // save the original data with centre point
    Mat m = im_short.getMat( im_short.SZ()/2 );
    save_slice( m, minVal, maxVal, "original_234_270.png", Vec2i(234, 270) );
    save_slice( m, minVal, maxVal, "original_233_269.png", Vec2i(233, 269) );


    Data3D<short> im_rduct, im_rduct2;
    vector<double> correction;

    RR::AccumulatePolarRD( im_short, im_rduct, RR::MED_DIFF, 0.2f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "polar_med_diff_v1.png" );

    RR::AccumulatePolarRD( im_short, im_rduct, RR::MED_DIFF,
                           0.2f, 233.6f, 269.6f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal,
                "polar_med_diff_v1_subpixel.png" );



    RR::polarRD( im_short, im_rduct, RR::AVG_DIFF,
                 1.0f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal,
                "polar_avg_diff.png" );

    RR::polarRD( im_short, im_rduct, RR::AVG_DIFF,
                 1.0f, 233.6f, 269.6f  );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal,
                "polar_avg_diff_subpixel.png" );

    RR::polarRD( im_short, im_rduct, RR::AVG_DIFF,
                 0.2f, 233.6f, 269.6f, 0.2f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal,
                "polar_avg_diff_subpixel_0.2gap.png" );




    RR::polarRD( im_short, im_rduct, RR::MED_DIFF, 1.0f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal,
                "polar_med_diff.png" );

    RR::polarRD( im_short, im_rduct, RR::MED_DIFF,
                 1.0f, 233.5f, 269.5f  );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal,
                "polar_med_diff_subpixel.png" );

    RR::polarRD( im_short, im_rduct, RR::MED_DIFF,
                 0.2f, 233.6f, 269.6f, 0.2f  );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal,
                "polar_med_diff_subpixel_0.2gap.png" );




    RR::sijbers( im_short, im_rduct, 1.0f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "sijbers.png" );

    RR::sijbers( im_short, im_rduct, 1.0f, 233.6f, 269.6f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "sijbers_subpixel.png" );

    RR::sijbers( im_short, im_rduct, 0.2f, 233.6f, 269.6f );
    save_slice( im_rduct, im_rduct.SZ()/2, minVal, maxVal, "sijbers_subpixel_0.2dr.png" );

    return 0;
}

