#include <iostream>
#include <opencv2/core/core.hpp>

#include "../core/Image3D.h"
#include "../core/GLViewerCore.h"
#include "../core/ImageProcessing.h"

using namespace std;
using namespace cv;


/*This project will generate a mask of the Aorta*/

int main()
{
    // TODO: This is for force the linking of OpenCV
    Mat m = Mat(1,1,CV_32F);
    imshow( "Temp", m );

    Image3D<short> im_short;
    bool flag = im_short.load( "../temp/vessel3d.data" );
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
    mask.save( "../temp/vessel3d.aorta.mask.data" );

    cout  << endl << endl << "Generating image with mask... ";
    cout.flush();
    Image3D<short> im_short2;
    IP::masking( im_short, mask, im_short2 );

    GLViewerCore vis;

    im_short.shrink_by_half();
    im_short2.shrink_by_half();
    mask.shrink_by_half();
    vis.addObject( im_short,  GLViewer::Volumn::MIP );
    vis.addObject( im_short2,  GLViewer::Volumn::MIP );
    vis.addObject( mask,  GLViewer::Volumn::MIP );


    vis.display( 1280, 800, 3 );

    return 0;
}
