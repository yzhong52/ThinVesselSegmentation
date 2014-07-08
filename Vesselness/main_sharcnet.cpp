#include <iostream>
#include "VesselDetector.h"
#include "Data3D.h"
#include "ImageProcessing.h"
#include "../send_email.h"

#define INPUT_DIR "./"
#define OUTPUT_DIR "./"

using namespace std;
using namespace cv;


namespace sample_code
{
// Compute vesselness measure
int vesselness( std::string dataname = "data15" );

// Extract Vessel centrelines with non-maximum suppression
int centreline( std::string dataname = "data15" );
}

int main(void)
{
    std::string dataname = "data15"; 
    sample_code::vesselness( dataname );
    sample_code::centreline( dataname );

    // Send an email to myself when it all done. 
    cout << "All done for vesselness. Sending the remail..." << endl; 
    send_email();
    return 0;
}

int sample_code::vesselness( string dataname )
{
    // Sigma: Parameters for Vesselness
    // [sigma_from, sigma_to]: the potential size rang of the vessels
    // sigma_step: precision of computation
    float sigma_from = 1.0f;
    float sigma_to   = 8.10f;
    float sigma_step = 0.3f;
    // Parameters for vesselness, please refer to Frangi's papaer
    // or this [blog](http://yzhong.co/?p=351)
    float alpha = 1.0e-1f;
    float beta  = 5.0e0f;
    float gamma = 3.5e5f;

    // Laoding data
    Image3D<short> im_short;
    bool flag = im_short.load( INPUT_DIR + dataname + ".data" );
    if( !flag ) return 0;

    // Compute Vesselness
    Data3D<Vesselness_Sig> vn_sig;
    VesselDetector::compute_vesselness( im_short, vn_sig, sigma_from, sigma_to, sigma_step,
                                        alpha, beta, gamma );

    // Saving the result
    vn_sig.save( OUTPUT_DIR + dataname + ".vn_sig" );

    return 0;
}

int sample_code::centreline( string dataname )
{
    // load vesselness data
    Data3D<Vesselness_Sig> vn_sig;
    vn_sig.load( OUTPUT_DIR + dataname + ".vn_sig" );

    // non-maximum suppression
    Data3D<Vesselness_Sig> vn_sig_nms;
    IP::non_max_suppress( vn_sig, vn_sig_nms );
    vn_sig_nms.save( OUTPUT_DIR + dataname + ".nms.vn_sig" );

    // edge tracing
    Data3D<Vesselness_Sig> vn_sig_et;
    IP::edge_tracing( vn_sig_nms, vn_sig_et, 0.45f, 0.05f );
    vn_sig_et.save( OUTPUT_DIR + dataname + ".et.vn_sig" );

    return 0;
}
