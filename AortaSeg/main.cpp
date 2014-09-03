#include <iostream>
#include <opencv2/core/core.hpp>

#include "../core/Image3D.h"
#include "../core/GLViewerCore.h"
#include "../core/ImageProcessing.h"

#include "../Vesselness/VesselnessTypes.h"

using namespace std;
using namespace cv;

/*This project will generate a mask of the Aorta*/

static const string dataname = "../temp/vessel3d_rd";
static const string maskname = "../temp/vessel3d.aorta.mask.data";

bool generating_segmentation(){
    Image3D<short> im_short;
    bool flag = im_short.load( dataname + ".data" );
    if( !flag ) return 0;

    cout << endl << endl << "Blur the data with Gaussian Filter... ";
    cout.flush();
    Data3D<short> im_short_blur;
    IP::GaussianBlur3D( im_short, im_short_blur, 61 );

    cout << endl << endl << "Generating a mask with thresholding... ";
    cout.flush();
    Image3D<unsigned char> mask;
    IP::threshold( im_short_blur, mask, short(6900) );

    cout << endl << endl << "Dialating the mask ...";
    cout.flush();
    IP::dilate( mask, 8 );
    mask.save( maskname );

    return true;
}

bool validating_segmentation(){
    bool flag = false;

    Image3D<unsigned char> im_mask;
    flag = im_mask.load( maskname );
    if( !flag ) return 0;
    im_mask.shrink_by_half();

    Image3D<short> im_short;
    flag = im_short.load( dataname + ".data" );
    if( !flag ) return 0;
    im_short.shrink_by_half();

    Image3D<Vesselness_Sig> vn_sig;
    vn_sig.load( dataname + ".et.vn_sig" );
    if( !flag ) return 0;
    Image3D<float> vn_float, vn_float2;
    vn_sig.copyDimTo(vn_float, 0 );
    vn_float.shrink_by_half();
    IP::masking( vn_float, im_mask, vn_float2 );

    cout << endl << endl << "Generating image with mask... ";
    cout.flush();
    Image3D<short> im_short2;
    IP::masking( im_short, im_mask, im_short2 );

    GLViewerCore vis;

    vis.addObject( im_short,  GLViewer::Volumn::MIP );
    vis.addObject( vn_float2,  GLViewer::Volumn::MIP );
    vis.addObject( im_short2,  GLViewer::Volumn::MIP );
    vis.addObject( im_mask,  GLViewer::Volumn::MIP );
    vis.display( 1280, 800, 4 );

    return true;
}

int main()
{
    // TODO: This is for force the linking of OpenCV
    Mat m = Mat(1,1,CV_32F);
    imshow( "Temp", m );

    //generating_segmentation();
    validating_segmentation();

    return 0;
}
