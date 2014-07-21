#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>

#include <opencv2/core/core.hpp>

// For computing vesselness
#include "VesselnessTypes.h"
#include "VesselDetector.h"

// For visualization
#include "GLViewerVesselness.h"

// Linux header: provide system call functions such as sleep()
#include <unistd.h>

using namespace std;
using namespace cv;

GLViewerVesselness viewer;

namespace sample_code
{
// Compute vesselness measure
int vesselness( bool isDisplay, std::string dataname = "data15", short threshold = 2900 );

// Extract Vessel centrelines with non-maximum suppression
int centreline( bool isDisplay, std::string dataname = "data15" );
}

int main(void)
{
    Mat temp = Mat(10, 10, CV_8UC3);
    cv::imshow( "", temp );

    /*
    Image3D<short> im_data;
    im_data.load( "../temp/vessel3d_rd_sp.data" );
    im_data.setROI( );
    im_data.saveROI( "../temp/roi16.data" );

    viewer.addObject( im_data.getROI(),  GLViewer::Volumn::MIP );
    viewer.display(600, 400, 1);

    return 0;
    /**/

    sample_code::vesselness(false, "../temp/roi16", 2800 );
    sample_code::centreline(true,  "../temp/roi16");
    return 0;
}

int sample_code::vesselness( bool isDisplay, string dataname, short threshold )
{
    // create output folders if it does not exist
    // CreateDirectory(L"../temp", NULL);

    // Sigma: Parameters for Vesselness
    // [sigma_from, sigma_to]: the potential size rang of the vessels
    // sigma_step: precision of computation
    float sigma_from = 0.65f;
    float sigma_to   = 10.10f;
    float sigma_step = 0.27f;
    // Parameters for vesselness, please refer to Frangi's papaer
    // or this [blog](http://yzhong.co/?p=351)
    float alpha = 0.3e0f;
    float beta  = 6.8e0f;
    float gamma = 4.5e5f; // Increase: small vessel disappear

    // Loading original data
    Image3D<short> im_short_orig;
    bool flag = im_short_orig.load( dataname + ".data" );
    if( !flag ) return 0;

    Data3D<short> im_short;
    IP::threshold( im_short_orig, im_short, threshold ); // [2500, 4500]

    // Compute Vesselness
    Data3D<Vesselness_Sig> vn_sig;
    VesselDetector::compute_vesselness( im_short, vn_sig,
                                        sigma_from, sigma_to, sigma_step,
                                        alpha, beta, gamma );
    vn_sig.save( dataname + ".vn_sig" );

    // If you want to visulize the data using Maximum-Intensity Projection
    if( isDisplay )
    {
        viewer.addObject( im_short_orig,  GLViewer::Volumn::MIP );
        viewer.addObject( im_short,  GLViewer::Volumn::MIP );
        viewer.addObject( vn_sig,  GLViewer::Volumn::MIP );
        viewer.addDiretionObject( vn_sig );
        viewer.display(800, 600, 4);
    }

    return 0;
}

int sample_code::centreline( bool isDisplay, string dataname )
{
    // load vesselness data
    Data3D<Vesselness_Sig> vn_sig( dataname + ".vn_sig" );

    // non-maximum suppression
    Data3D<Vesselness_Sig> vn_sig_nms;
    IP::non_max_suppress( vn_sig, vn_sig_nms );
    vn_sig_nms.save( dataname + ".nms.vn_sig" );

    // edge tracing
    Data3D<Vesselness_Sig> vn_sig_et;
    IP::edge_tracing( vn_sig_nms, vn_sig_et, 0.05f, 0.01f );
    vn_sig_et.save( dataname + ".et.vn_sig" );

    // Loading original data
    Data3D<short> im_short_orig;
    bool flag = im_short_orig.load( dataname + ".data" );
    if( !flag ) return 0;

    if( isDisplay )
    {
        viewer.addObject( im_short_orig );
        viewer.addObject( vn_sig );
        viewer.addObject( vn_sig_et,  GLViewer::Volumn::MIP );
        viewer.addDiretionObject( vn_sig_et );
        viewer.display(800, 600, 4);
    }
    return 0;
}



